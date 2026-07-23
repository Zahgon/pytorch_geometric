from typing import Union

from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import scatter, spmm


class WLConvContinuous(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr='add', **kwargs)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        if isinstance(edge_index, SparseTensor):
            assert edge_weight is None
            dst_index, _, edge_weight = edge_index.coo()
        else:
            dst_index = edge_index[1]

        if edge_weight is None:
            edge_weight = x[0].new_ones(dst_index.numel())

        deg = scatter(edge_weight, dst_index, 0, out.size(0), reduce='sum')
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        out = deg_inv.view(-1, 1) * out

        x_dst = x[1]
        if x_dst is not None:
            out = 0.5 * (x_dst + out)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)
