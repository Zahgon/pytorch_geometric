import random
from typing import Tuple

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random_scale')
class RandomScale(BaseTransform):
    def __init__(self, scales: Tuple[float, float]) -> None:
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        scale = random.uniform(*self.scales)
        data.pos = data.pos * scale
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.scales})'
