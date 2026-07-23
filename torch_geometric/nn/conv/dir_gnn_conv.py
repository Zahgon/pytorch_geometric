import copy

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing


class DirGNNConv(torch.nn.Module):
    def __init__(
        self,
        conv: MessagePassing,
        alpha: float = 0.5,
        root_weight: bool = True,
    ):
        super().__init__()

        self.alpha = alpha
        self.root_weight = root_weight

        self.conv_in = copy.deepcopy(conv)
        self.conv_out = copy.deepcopy(conv)

        if hasattr(conv, 'add_self_loops'):
            self.conv_in.add_self_loops = False
            self.conv_out.add_self_loops = False
        if hasattr(conv, 'root_weight'):
            self.conv_in.root_weight = False
            self.conv_out.root_weight = False

        if root_weight:
            self.lin = torch.nn.Linear(conv.in_channels, conv.out_channels)
        else:
            self.lin = None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv_in.reset_parameters()
        self.conv_out.reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """"""  # noqa: D419
        x_in = self.conv_in(x, edge_index)
        x_out = self.conv_out(x, edge_index.flip([0]))

        out = self.alpha * x_out + (1 - self.alpha) * x_in

        if self.root_weight:
            out = out + self.lin(x)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.conv_in}, alpha={self.alpha})'
