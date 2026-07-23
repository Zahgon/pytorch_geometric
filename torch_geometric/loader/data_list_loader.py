from typing import List, Union

import torch

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData


def collate_fn(data_list):
    pass


class DataListLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Union[Dataset, List[BaseData]],
                 batch_size: int = 1, shuffle: bool = False, **kwargs):
        kwargs.pop('collate_fn', None)

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, **kwargs)
