import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs, read_planetoid_data


class Planetoid(InMemoryDataset):
    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    geom_gcn_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                    'geom-gcn/master')

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "public",
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name

        self.split = split.lower()
        assert self.split in ['public', 'full', 'geom-gcn', 'random']

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        pass

    @property
    def processed_dir(self) -> str:
        pass

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        for name in self.raw_file_names:
            fs.cp(f'{self.url}/{name}', self.raw_dir)
        if self.split == 'geom-gcn':
            for i in range(10):
                url = f'{self.geom_gcn_url}/splits/{self.name.lower()}'
                fs.cp(f'{url}_split_0.6_0.2_{i}.npz', self.raw_dir)

    def process(self) -> None:
        data = read_planetoid_data(self.raw_dir, self.name)

        if self.split == 'geom-gcn':
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f'{self.name.lower()}_split_0.6_0.2_{i}.npz'
                splits = np.load(osp.join(self.raw_dir, name))
                train_masks.append(torch.from_numpy(splits['train_mask']))
                val_masks.append(torch.from_numpy(splits['val_mask']))
                test_masks.append(torch.from_numpy(splits['test_mask']))
            data.train_mask = torch.stack(train_masks, dim=1)
            data.val_mask = torch.stack(val_masks, dim=1)
            data.test_mask = torch.stack(test_masks, dim=1)

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
