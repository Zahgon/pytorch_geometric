import math
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import GCNConv, MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj


class AntiSymmetricConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        phi: Optional[MessagePassing] = None,
        num_iters: int = 1,
        epsilon: float = 0.1,
        gamma: float = 0.1,
        act: Union[str, Callable, None] = 'tanh',
        act_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.act = activation_resolver(act, **(act_kwargs or {}))

        if phi is None:
            phi = GCNConv(in_channels, in_channels, bias=False)

        self.W = Parameter(torch.empty(in_channels, in_channels))
        self.register_buffer('eye', torch.eye(in_channels))
        self.phi = phi

        if bias:
            self.bias = Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.phi.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        r"""Runs the forward pass of the module."""
        antisymmetric_W = self.W - self.W.t() - self.gamma * self.eye

        for _ in range(self.num_iters):
            h = self.phi(x, edge_index, *args, **kwargs)
            h = x @ antisymmetric_W.t() + h

            if self.bias is not None:
                h += self.bias

            if self.act is not None:
                h = self.act(h)

            x = x + self.epsilon * h

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.in_channels}, '
                f'phi={self.phi}, '
                f'num_iters={self.num_iters}, '
                f'epsilon={self.epsilon}, '
                f'gamma={self.gamma})')
