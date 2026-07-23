from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import SimpleConv
from torch_geometric.nn.dense.linear import Linear


class PMLP(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.,
        norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels, self.bias))
        for _ in range(self.num_layers - 2):
            lin = Linear(hidden_channels, hidden_channels, self.bias)
            self.lins.append(lin)
        self.lins.append(Linear(hidden_channels, out_channels, self.bias))

        self.norm = None
        if norm:
            self.norm = torch.nn.BatchNorm1d(
                hidden_channels,
                affine=False,
                track_running_stats=False,
            )

        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            torch.nn.init.xavier_uniform_(lin.weight, gain=1.414)
            if self.bias:
                torch.nn.init.zeros_(lin.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[Tensor] = None,
    ) -> torch.Tensor:
        """"""  # noqa: D419
        if not self.training and edge_index is None:
            raise ValueError(f"'edge_index' needs to be present during "
                             f"inference in '{self.__class__.__name__}'")

        for i in range(self.num_layers):
            x = x @ self.lins[i].weight.t()
            if not self.training:
                x = self.conv(x, edge_index)
            if self.bias:
                x = x + self.lins[i].bias
            if i != self.num_layers - 1:
                if self.norm is not None:
                    x = self.norm(x)
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
