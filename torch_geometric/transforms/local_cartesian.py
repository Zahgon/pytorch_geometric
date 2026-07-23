from typing import Tuple

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter


@functional_transform('local_cartesian')
class LocalCartesian(BaseTransform):
    def __init__(
            self,
            norm: bool = True,
            cat: bool = True,
            interval: Tuple[float, float] = (0.0, 1.0),
    ):
        self.norm = norm
        self.cat = cat
        self.interval = interval

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[row] - pos[col]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if self.norm:
            max_value = scatter(cart.abs(), col, 0, pos.size(0), reduce='max')
            max_value = max_value.max(dim=-1, keepdim=True)[0]

            length = self.interval[1] - self.interval[0]
            center = (self.interval[0] + self.interval[1]) / 2
            cart = length * cart / (2 * max_value[col]) + center

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data
