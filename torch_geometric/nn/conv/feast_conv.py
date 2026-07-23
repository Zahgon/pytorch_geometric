from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import normal
from torch_geometric.typing import Adj, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops


class FeaStConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='uniform')
        self.u = Linear(in_channels, heads, bias=False,
                        weight_initializer='uniform')
        self.c = Parameter(torch.empty(heads))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.u.reset_parameters()
        normal(self.c, mean=0, std=0.1)
        normal(self.bias, mean=0, std=0.1)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=x[1].size(0))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        out = self.propagate(edge_index, x=x)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        q = self.u(x_j - x_i) + self.c  # Translation invariance.
        q = F.softmax(q, dim=1)
        x_j = self.lin(x_j).view(x_j.size(0), self.heads, -1)
        return (x_j * q.view(-1, self.heads, 1)).sum(dim=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
