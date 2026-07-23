import typing
from typing import Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import PairTensor  # noqa
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    SparseTensor,
)
from torch_geometric.utils import is_torch_sparse_tensor
from torch_geometric.utils.sparse import set_sparse_value

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


class FAConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]
    _alpha: OptTensor

    def __init__(self, channels: int, eps: float = 0.1, dropout: float = 0.0,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.channels = channels
        self.eps = eps
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._alpha = None

        self.att_l = Linear(channels, 1, bias=False)
        self.att_r = Linear(channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.att_l.reset_parameters()
        self.att_r.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    @overload
    def forward(
        self,
        x: Tensor,
        x_0: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Tensor,
        x_0: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Tensor,
        x_0: Tensor,
        edge_index: SparseTensor,
        edge_weight: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Tensor,
        x_0: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The node features.
            x_0 (torch.Tensor): The initial input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights.
                (default: :obj:`None`)
            return_attention_weights (bool, optional):
                Will additionally return the tuple
                :obj:`(edge_index, attention_weights)` whenever it is set to
                a value, regardless of its actual value
                (might be `True` or `False`), holding the computed attention
                weights for each edge.
                (default: :obj:`None`)
        """
        if self.normalize:
            if isinstance(edge_index, Tensor):
                assert edge_weight is None
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, None, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                assert not edge_index.has_value()
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, None, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        else:
            if isinstance(edge_index,
                          Tensor) and not is_torch_sparse_tensor(edge_index):
                assert edge_weight is not None
            elif isinstance(edge_index, SparseTensor):
                assert edge_index.has_value()

        alpha_l = self.att_l(x)
        alpha_r = self.att_r(x)

        out = self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r),
                             edge_weight=edge_weight)

        alpha = self._alpha
        self._alpha = None

        if self.eps != 0.0:
            out = out + self.eps * x_0

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor,
                edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        alpha = (alpha_j + alpha_i).tanh().squeeze(-1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * (alpha * edge_weight).view(-1, 1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, eps={self.eps})'
