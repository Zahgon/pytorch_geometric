import glob
import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import read_off


class GeometricShapes(InMemoryDataset):

    url = 'https://github.com/Yannick-S/geometric_shapes/raw/master/raw.zip'

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)

    @property
    def raw_file_names(self) -> str:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)

    def process(self) -> None:
        self.save(self.process_set('train'), self.processed_paths[0])
        self.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset: str) -> List[Data]:
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob(f'{folder}/*.off')
            for path in paths:
                data = read_off(path)
                assert data.pos is not None
                data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return data_list
