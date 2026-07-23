from typing import Optional

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear


class DiffGroupNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        groups: int,
        lamda: float = 0.01,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.groups = groups
        self.lamda = lamda

        self.lin = Linear(in_channels, groups, bias=False, device=device)
        self.norm = BatchNorm1d(groups * in_channels, eps, momentum, affine,
                                track_running_stats, device=device)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
        """
        F, G = self.in_channels, self.groups

        s = self.lin(x).softmax(dim=-1)  # [N, G]
        out = s.unsqueeze(-1) * x.unsqueeze(-2)  # [N, G, F]
        out = self.norm(out.view(-1, G * F)).view(-1, G, F).sum(-2)  # [N, F]

        return x + self.lamda * out

    @staticmethod
    def group_distance_ratio(x: Tensor, y: Tensor, eps: float = 1e-5) -> float:
        pass

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'groups={self.groups})')
