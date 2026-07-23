from math import ceil, log2
from typing import Optional

import torch
from torch import Tensor
from torch.nn import GRUCell, Linear

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation


class LCMAggregation(Aggregation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        project: bool = True,
    ):
        super().__init__()

        if in_channels != out_channels and not project:
            raise ValueError(f"Inputs of '{self.__class__.__name__}' must be "
                             f"projected if `in_channels != out_channels`")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.project = project

        if self.project:
            self.lin = Linear(in_channels, out_channels)
        else:
            self.lin = None

        self.gru_cell = GRUCell(out_channels, out_channels)

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.gru_cell.reset_parameters()

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:

        if self.project:
            x = self.lin(x).relu()

        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=max_num_elements)

        x = x.permute(1, 0, 2)  # [num_neighbors, num_nodes, num_features]
        _, num_nodes, num_features = x.size()

        depth = ceil(log2(x.size(0)))
        for _ in range(depth):
            half_size = ceil(x.size(0) / 2)

            if x.size(0) % 2 == 1:
                x, remainder = x[:-1], x[-1:]
            else:
                remainder = None

            left_right = x.view(-1, 2, num_nodes, num_features)
            right_left = left_right.flip(dims=[1])

            left_right = left_right.reshape(-1, num_features)
            right_left = right_left.reshape(-1, num_features)

            out = self.gru_cell(left_right, right_left)
            out = out.view(-1, 2, num_nodes, num_features)
            out = out.mean(dim=1)
            if remainder is not None:
                out = torch.cat([out, remainder], dim=0)

            x = out.view(half_size, num_nodes, num_features)

        assert x.size(0) == 1
        return x.squeeze(0)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, project={self.project})')
