from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class LGConv(MessagePassing):
    def __init__(self, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.normalize = normalize

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize and isinstance(edge_index, Tensor):
            out = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                           add_self_loops=False, flow=self.flow, dtype=x.dtype)
            edge_index, edge_weight = out
        elif self.normalize and isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim),
                                  add_self_loops=False, flow=self.flow,
                                  dtype=x.dtype)

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
