from typing import List, Union

import torch
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Data, Dataset


def collate_fn(data_list: List[Data]) -> Batch:
    pass


class DenseDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Union[Dataset, List[Data]],
                 batch_size: int = 1, shuffle: bool = False, **kwargs):
        kwargs.pop('collate_fn', None)

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, **kwargs)
