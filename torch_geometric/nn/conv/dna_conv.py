import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import kaiming_uniform, uniform
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor


class Linear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.weight = Parameter(
            torch.empty(groups, in_channels // groups, out_channels // groups))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.weight, fan=self.weight.size(1), a=math.sqrt(5))
        uniform(self.weight.size(1), self.bias)

    def forward(self, src):

        if self.groups > 1:
            size = src.size()[:-1]
            src = src.view(-1, self.groups, self.in_channels // self.groups)
            src = src.transpose(0, 1).contiguous()
            out = torch.matmul(src, self.weight)
            out = out.transpose(1, 0).contiguous()
            out = out.view(size + (self.out_channels, ))
        else:
            out = torch.matmul(src, self.weight.squeeze(0))

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:  # pragma: no cover
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, groups={self.groups})')


def restricted_softmax(src, dim: int = -1, margin: float = 0.):
    src_max = torch.clamp(src.max(dim=dim, keepdim=True)[0], min=0.)
    out = (src - src_max).exp()
    out = out / (out.sum(dim=dim, keepdim=True) + (margin - src_max).exp())
    return out


class Attention(torch.nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = dropout

    def forward(self, query, key, value):
        return self.compute_attention(query, key, value)

    def compute_attention(self, query, key, value):

        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1)
        assert key.size(-2) == value.size(-2)

        score = torch.matmul(query, key.transpose(-2, -1))
        score = score / math.sqrt(key.size(-1))
        score = restricted_softmax(score, dim=-1)
        score = F.dropout(score, p=self.dropout, training=self.training)

        return torch.matmul(score, value)

    def __repr__(self) -> str:  # pragma: no cover
        return f'{self.__class__.__name__}(dropout={self.dropout})'


class MultiHead(Attention):
    def __init__(self, in_channels, out_channels, heads=1, groups=1, dropout=0,
                 bias=True):
        super().__init__(dropout)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.groups = groups
        self.bias = bias

        assert in_channels % heads == 0 and out_channels % heads == 0
        assert in_channels % groups == 0 and out_channels % groups == 0
        assert max(groups, self.heads) % min(groups, self.heads) == 0

        self.lin_q = Linear(in_channels, out_channels, groups, bias)
        self.lin_k = Linear(in_channels, out_channels, groups, bias)
        self.lin_v = Linear(in_channels, out_channels, groups, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()

    def forward(self, query, key, value):

        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1) == value.size(-1)
        assert key.size(-2) == value.size(-2)

        query = self.lin_q(query)
        key = self.lin_k(key)
        value = self.lin_v(value)

        size = query.size()[:-2]
        out_channels_per_head = self.out_channels // self.heads

        query_size = size + (query.size(-2), self.heads, out_channels_per_head)
        query = query.view(query_size).transpose(-2, -3)

        key_size = size + (key.size(-2), self.heads, out_channels_per_head)
        key = key.view(key_size).transpose(-2, -3)

        value_size = size + (value.size(-2), self.heads, out_channels_per_head)
        value = value.view(value_size).transpose(-2, -3)

        out = self.compute_attention(query, key, value)
        out = out.transpose(-3, -2).contiguous()
        out = out.view(size + (query.size(-2), self.out_channels))

        return out

    def __repr__(self) -> str:  # pragma: no cover
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'groups={self.groups}, dropout={self.dropout}, '
                f'bias={self.bias})')


class DNAConv(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, heads: int = 1, groups: int = 1,
                 dropout: float = 0., cached: bool = False,
                 normalize: bool = True, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.bias = bias
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.multi_head = MultiHead(channels, channels, heads, groups, dropout,
                                    bias)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.multi_head.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The input node features of shape
                :obj:`[num_nodes, num_layers, channels]`.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights.
                (default: :obj:`None`)
        """
        if x.dim() != 3:
            raise ValueError('Feature shape must be [num_nodes, num_layers, '
                             'channels].')

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        x_i = x_i[:, -1:]  # [num_edges, 1, channels]
        out = self.multi_head(x_i, x_j, x_j)  # [num_edges, 1, channels]
        return edge_weight.view(-1, 1) * out.squeeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.multi_head.in_channels}, '
                f'heads={self.multi_head.heads}, '
                f'groups={self.multi_head.groups})')
