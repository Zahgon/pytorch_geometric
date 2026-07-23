import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj


class PointGNNConv(MessagePassing):
    def __init__(
        self,
        mlp_h: torch.nn.Module,
        mlp_f: torch.nn.Module,
        mlp_g: torch.nn.Module,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.mlp_h = mlp_h
        self.mlp_f = mlp_f
        self.mlp_g = mlp_g

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_h)
        reset(self.mlp_f)
        reset(self.mlp_g)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj) -> Tensor:
        out = self.propagate(edge_index, x=x, pos=pos)
        out = self.mlp_g(out)
        return x + out

    def message(self, pos_j: Tensor, pos_i: Tensor, x_i: Tensor,
                x_j: Tensor) -> Tensor:
        delta = self.mlp_h(x_i)
        e = torch.cat([pos_j - pos_i + delta, x_j], dim=-1)
        return self.mlp_f(e)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  mlp_h={self.mlp_h},\n'
                f'  mlp_f={self.mlp_f},\n'
                f'  mlp_g={self.mlp_g},\n'
                f')')
