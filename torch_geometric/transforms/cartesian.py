from typing import Optional, Tuple

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('cartesian')
class Cartesian(BaseTransform):
    def __init__(
            self,
            norm: bool = True,
            max_value: Optional[float] = None,
            cat: bool = True,
            interval: Tuple[float, float] = (0.0, 1.0),
    ):
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.interval = interval

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[row] - pos[col]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if self.norm and cart.numel() > 0:
            max_val = float(cart.abs().max()) if self.max is None else self.max

            length = self.interval[1] - self.interval[0]
            center = (self.interval[0] + self.interval[1]) / 2
            cart = length * cart / (2 * max_val) + center

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
