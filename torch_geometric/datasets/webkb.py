import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce


class WebKB(InMemoryDataset):

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'wisconsin']

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

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
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0]) as f:
            lines = f.read().split('\n')[1:-1]
            xs = [[float(value) for value in line.split('\t')[1].split(',')]
                  for line in lines]
            x = torch.tensor(xs, dtype=torch.float)

            ys = [int(line.split('\t')[2]) for line in lines]
            y = torch.tensor(ys, dtype=torch.long)

        with open(self.raw_paths[1]) as f:
            lines = f.read().split('\n')[1:-1]
            edge_indices = [[int(value) for value in line.split('\t')]
                            for line in lines]
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_index = coalesce(edge_index, num_nodes=x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for path in self.raw_paths[2:]:
            tmp = np.load(path)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
