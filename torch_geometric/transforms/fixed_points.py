import math
import re

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('fixed_points')
class FixedPoints(BaseTransform):
    def __init__(
        self,
        num: int,
        replace: bool = True,
        allow_duplicates: bool = False,
    ):
        self.num = num
        self.replace = replace
        self.allow_duplicates = allow_duplicates

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        assert num_nodes is not None

        if self.replace:
            choice = torch.from_numpy(
                np.random.choice(num_nodes, self.num, replace=True)).long()
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes)[:self.num]
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        for key, value in data.items():
            if key == 'num_nodes':
                data.num_nodes = choice.size(0)
            elif bool(re.search('edge', key)):
                continue
            elif (isinstance(value, Tensor) and value.size(0) == num_nodes
                  and value.size(0) != 1):
                data[key] = value[choice]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num}, replace={self.replace})'
