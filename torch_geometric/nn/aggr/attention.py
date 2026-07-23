from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax


class AttentionalAggregation(Aggregation):
    def __init__(
        self,
        gate_nn: torch.nn.Module,
        nn: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        from torch_geometric.nn import MLP

        self.gate_nn = self.gate_mlp = None
        if isinstance(gate_nn, MLP):
            self.gate_mlp = gate_nn
        else:
            self.gate_nn = gate_nn

        self.nn = self.mlp = None
        if isinstance(nn, MLP):
            self.mlp = nn
        else:
            self.nn = nn

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.gate_mlp)
        reset(self.nn)
        reset(self.mlp)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        if self.gate_mlp is not None:
            gate = self.gate_mlp(x, batch=index, batch_size=dim_size)
        else:
            gate = self.gate_nn(x)

        if self.mlp is not None:
            x = self.mlp(x, batch=index, batch_size=dim_size)
        elif self.nn is not None:
            x = self.nn(x)

        gate = softmax(gate, index, ptr, dim_size, dim)
        return self.reduce(gate * x, index, ptr, dim_size, dim)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'gate_nn={self.gate_mlp or self.gate_nn}, '
                f'nn={self.mlp or self.nn})')
