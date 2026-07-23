import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor, torch_sparse
from torch_geometric.utils import (
    add_self_loops,
    degree,
    is_torch_sparse_tensor,
    remove_self_loops,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.sparse import set_sparse_value


class ClusterGCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 diag_lambda: float = 0., add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diag_lambda = diag_lambda
        self.add_self_loops = add_self_loops

        self.lin_out = Linear(in_channels, out_channels, bias=bias,
                              weight_initializer='glorot')
        self.lin_root = Linear(in_channels, out_channels, bias=False,
                               weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_out.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        num_nodes = x.size(self.node_dim)
        edge_weight: OptTensor = None

        if isinstance(edge_index, SparseTensor):
            assert edge_index.size(0) == edge_index.size(1)

            if self.add_self_loops:
                edge_index = torch_sparse.set_diag(edge_index)

            col, row, _ = edge_index.coo()  # Transposed.
            deg_inv = 1. / torch_sparse.sum(edge_index, dim=1).clamp_(1.)

            edge_weight = deg_inv[col]
            edge_weight[row == col] += self.diag_lambda * deg_inv
            edge_index = edge_index.set_value(edge_weight, layout='coo')

        elif is_torch_sparse_tensor(edge_index):
            assert edge_index.size(0) == edge_index.size(1)

            if edge_index.layout == torch.sparse_csc:
                raise NotImplementedError("Sparse CSC matrices are not yet "
                                          "supported in 'gcn_norm'")

            if self.add_self_loops:
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

            col_and_row, value = to_edge_index(edge_index)
            col, row = col_and_row[0], col_and_row[1]
            deg_inv = 1. / degree(col, num_nodes=edge_index.size(0)).clamp_(1.)

            edge_weight = deg_inv[col]
            edge_weight[row == col] += self.diag_lambda * deg_inv

            edge_index = set_sparse_value(edge_index, edge_weight)

        else:
            if self.add_self_loops:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

            row, col = edge_index[0], edge_index[1]
            deg_inv = 1. / degree(col, num_nodes=num_nodes).clamp_(1.)

            edge_weight = deg_inv[col]
            edge_weight[row == col] += self.diag_lambda * deg_inv

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_out(out) + self.lin_root(x)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, diag_lambda={self.diag_lambda})')
