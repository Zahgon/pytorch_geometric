from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_self_loops, spmm


class SimpleConv(MessagePassing):
    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = "sum",
        combine_root: Optional[str] = None,
        **kwargs,
    ):
        if combine_root not in ['sum', 'cat', 'self_loop', None]:
            raise ValueError(f"Received invalid value for 'combine_root' "
                             f"(got '{combine_root}')")

        super().__init__(aggr, **kwargs)
        self.combine_root = combine_root

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:

        if self.combine_root is not None:
            if self.combine_root == 'self_loop':
                if not isinstance(x, Tensor) or (size is not None
                                                 and size[0] != size[1]):
                    raise ValueError("Cannot use `combine_root='self_loop'` "
                                     "for bipartite message passing")
                if isinstance(edge_index, Tensor):
                    edge_index, edge_weight = add_self_loops(
                        edge_index, edge_weight, num_nodes=x.size(0))
                elif isinstance(edge_index, SparseTensor):
                    edge_index = torch_sparse.set_diag(edge_index)

        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        x_dst = x[1]
        if x_dst is not None and self.combine_root is not None:
            if self.combine_root == 'sum':
                out = out + x_dst
            elif self.combine_root == 'cat':
                out = torch.cat([x_dst, out], dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        assert isinstance(self.aggr, str)
        return spmm(adj_t, x[0], reduce=self.aggr)
