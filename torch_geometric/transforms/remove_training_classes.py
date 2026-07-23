from typing import List

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('remove_training_classes')
class RemoveTrainingClasses(BaseTransform):
    def __init__(self, classes: List[int]):
        self.classes = classes

    def forward(self, data: Data) -> Data:
        data.train_mask = data.train_mask.clone()
        for i in self.classes:
            data.train_mask[data.y == i] = False
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.classes})'
