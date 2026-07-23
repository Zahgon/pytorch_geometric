from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.aggr.fused import FusedAggregation


class BatchNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        allow_single_element: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if allow_single_element and not track_running_stats:
            raise ValueError("'allow_single_element' requires "
                             "'track_running_stats' to be set to `True`")

        self.module = torch.nn.BatchNorm1d(in_channels, eps, momentum, affine,
                                           track_running_stats, device=device)
        self.in_channels = in_channels
        self.allow_single_element = allow_single_element

    def reset_running_stats(self):
        r"""Resets all running statistics of the module."""
        self.module.reset_running_stats()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
        """
        if self.allow_single_element and x.size(0) <= 1:
            return torch.nn.functional.batch_norm(
                x,
                self.module.running_mean,
                self.module.running_var,
                self.module.weight,
                self.module.bias,
                False,  # bn_training
                0.0,  # momentum
                self.module.eps,
            )
        return self.module(x)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.module.extra_repr()})'


class HeteroBatchNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_types: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_types = num_types
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(
                torch.empty(num_types, in_channels, device=device))
            self.bias = Parameter(
                torch.empty(num_types, in_channels, device=device))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer(
                'running_mean',
                torch.empty(num_types, in_channels, device=device))
            self.register_buffer(
                'running_var',
                torch.empty(num_types, in_channels, device=device))
            self.register_buffer('num_batches_tracked', torch.tensor(0))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.mean_var = FusedAggregation(['mean', 'var'])

        self.reset_parameters()

    def reset_running_stats(self):
        r"""Resets all running statistics of the module."""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, type_vec: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input features.
            type_vec (torch.Tensor): A vector that maps each entry to a type.
        """
        if not self.training and self.track_running_stats:
            mean, var = self.running_mean, self.running_var
        else:
            with torch.no_grad():
                mean, var = self.mean_var(x, type_vec, dim_size=self.num_types)

        if self.training and self.track_running_stats:
            if self.momentum is None:
                self.num_batches_tracked.add_(1)
                exp_avg_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exp_avg_factor = self.momentum

            with torch.no_grad():  # Update running mean and variance:
                type_index = torch.unique(type_vec)

                self.running_mean[type_index] = (
                    (1.0 - exp_avg_factor) * self.running_mean[type_index] +
                    exp_avg_factor * mean[type_index])
                self.running_var[type_index] = (
                    (1.0 - exp_avg_factor) * self.running_var[type_index] +
                    exp_avg_factor * var[type_index])

        out = (x - mean[type_vec]) / var.clamp(self.eps).sqrt()[type_vec]

        if self.affine:
            out = out * self.weight[type_vec] + self.bias[type_vec]

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'num_types={self.num_types})')
