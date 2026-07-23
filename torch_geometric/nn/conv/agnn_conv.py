from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class AGNNConv(MessagePassing):
    def __init__(self, requires_grad: bool = True, add_self_loops: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.requires_grad = requires_grad
        self.add_self_loops = add_self_loops

        if requires_grad:
            self.beta = Parameter(torch.empty(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.requires_grad:
            self.beta.data.fill_(1)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=x.size(self.node_dim))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        x_norm = F.normalize(x, p=2., dim=-1)

        return self.propagate(edge_index, x=x, x_norm=x_norm)

    def message(self, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = self.beta * (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha.view(-1, 1)
