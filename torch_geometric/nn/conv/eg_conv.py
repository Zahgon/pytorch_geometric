from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_remaining_self_loops, scatter, spmm


class EGConv(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, OptTensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: Optional[List[str]] = None,
        num_heads: int = 8,
        num_bases: int = 4,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        if out_channels % num_heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads "
                             f"(got {num_heads})")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.aggregators = aggregators or ['symnorm']

        for a in self.aggregators:
            if a not in ['sum', 'mean', 'symnorm', 'min', 'max', 'var', 'std']:
                raise ValueError(f"Unsupported aggregator: '{a}'")

        self.bases_lin = Linear(in_channels,
                                (out_channels // num_heads) * num_bases,
                                bias=False, weight_initializer='glorot')
        self.comb_lin = Linear(in_channels,
                               num_heads * num_bases * len(self.aggregators))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.bases_lin.reset_parameters()
        self.comb_lin.reset_parameters()
        zeros(self.bias)
        self._cached_adj_t = None
        self._cached_edge_index = None

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        symnorm_weight: OptTensor = None
        if "symnorm" in self.aggregators:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, symnorm_weight = gcn_norm(  # yapf: disable
                        edge_index, None, num_nodes=x.size(self.node_dim),
                        improved=False, add_self_loops=self.add_self_loops,
                        flow=self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, symnorm_weight)
                else:
                    edge_index, symnorm_weight = cache

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, None, num_nodes=x.size(self.node_dim),
                        improved=False, add_self_loops=self.add_self_loops,
                        flow=self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        elif self.add_self_loops:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if self.cached and cache is not None:
                    edge_index = cache[0]
                else:
                    edge_index, _ = add_remaining_self_loops(edge_index)
                    if self.cached:
                        self._cached_edge_index = (edge_index, None)

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if self.cached and cache is not None:
                    edge_index = cache
                else:
                    edge_index = torch_sparse.fill_diag(edge_index, 1.0)
                    if self.cached:
                        self._cached_adj_t = edge_index

        bases = self.bases_lin(x)
        weightings = self.comb_lin(x)

        aggregated = self.propagate(edge_index, x=bases,
                                    symnorm_weight=symnorm_weight)

        weightings = weightings.view(-1, self.num_heads,
                                     self.num_bases * len(self.aggregators))
        aggregated = aggregated.view(
            -1,
            len(self.aggregators) * self.num_bases,
            self.out_channels // self.num_heads,
        )

        out = torch.matmul(weightings, aggregated)
        out = out.view(-1, self.out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None,
                  symnorm_weight: OptTensor = None) -> Tensor:

        outs = []
        for aggr in self.aggregators:
            if aggr == 'symnorm':
                assert symnorm_weight is not None
                out = scatter(inputs * symnorm_weight.view(-1, 1), index, 0,
                              dim_size, reduce='sum')
            elif aggr == 'var' or aggr == 'std':
                mean = scatter(inputs, index, 0, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, dim_size,
                                       reduce='mean')
                out = mean_squares - mean * mean
                if aggr == 'std':
                    out = out.clamp(min=1e-5).sqrt()
            else:
                out = scatter(inputs, index, 0, dim_size, reduce=aggr)

            outs.append(out)

        return torch.stack(outs, dim=1) if len(outs) > 1 else outs[0]

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        adj_t_2 = adj_t
        if len(self.aggregators) > 1 and 'symnorm' in self.aggregators:
            if isinstance(adj_t, SparseTensor):
                adj_t_2 = adj_t.set_value(None)
            else:
                adj_t_2 = adj_t.clone()
                adj_t_2.values().fill_(1.0)

        outs = []
        for aggr in self.aggregators:
            if aggr == 'symnorm':
                out = spmm(adj_t, x, reduce='sum')
            elif aggr in ['var', 'std']:
                mean = spmm(adj_t_2, x, reduce='mean')
                mean_sq = spmm(adj_t_2, x * x, reduce='mean')
                out = mean_sq - mean * mean
                if aggr == 'std':
                    out = torch.sqrt(out.relu_() + 1e-5)
            else:
                out = spmm(adj_t_2, x, reduce=aggr)

            outs.append(out)

        return torch.stack(outs, dim=1) if len(outs) > 1 else outs[0]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggregators={self.aggregators})')
