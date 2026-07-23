from typing import Optional

from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class SSGConv(MessagePassing):

    _cached_h: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, alpha: float,
                 K: int = 1, cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_h = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_h = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_h
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)

            h = x * self.alpha
            for _ in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
                h = h + (1 - self.alpha) / self.K * x
            if self.cached:
                self._cached_h = h
        else:
            h = cache.detach()

        return self.lin(h)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, alpha={self.alpha})')
