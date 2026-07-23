from typing import Union

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, LinearTransformation


@functional_transform('random_shear')
class RandomShear(BaseTransform):
    def __init__(self, shear: Union[float, int]) -> None:
        self.shear = abs(shear)

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        dim = data.pos.size(-1)

        matrix = data.pos.new_empty(dim, dim).uniform_(-self.shear, self.shear)
        eye = torch.arange(dim, dtype=torch.long)
        matrix[eye, eye] = 1

        return LinearTransformation(matrix)(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.shear})'
