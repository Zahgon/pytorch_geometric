from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax


class Set2Set(Aggregation):
    def __init__(self, in_channels: int, processing_steps: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.lstm = torch.nn.LSTM(self.out_channels, in_channels, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        self.assert_index_present(index)
        self.assert_two_dimensional_input(x, dim)

        h = (x.new_zeros((self.lstm.num_layers, dim_size, x.size(-1))),
             x.new_zeros((self.lstm.num_layers, dim_size, x.size(-1))))
        q_star = x.new_zeros(dim_size, self.out_channels)

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(dim_size, self.in_channels)
            e = (x * q[index]).sum(dim=-1, keepdim=True)
            a = softmax(e, index, ptr, dim_size, dim)
            r = self.reduce(a * x, index, ptr, dim_size, dim, reduce='sum')
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
