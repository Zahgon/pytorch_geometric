import random

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random_flip')
class RandomFlip(BaseTransform):
    def __init__(self, axis: int, p: float = 0.5) -> None:
        self.axis = axis
        self.p = p

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        if random.random() < self.p:
            pos = data.pos.clone()
            pos[..., self.axis] = -pos[..., self.axis]
            data.pos = pos
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(axis={self.axis}, p={self.p})'
