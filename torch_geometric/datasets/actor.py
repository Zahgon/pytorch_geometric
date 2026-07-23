from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce


class Actor(InMemoryDataset):

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0]) as f:
            node_data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, line, _ in node_data:
                indices = [int(x) for x in line.split(',')]
                rows += [int(n_id)] * len(indices)
                cols += indices
            row, col = torch.tensor(rows), torch.tensor(cols)

            x = torch.zeros(int(row.max()) + 1, int(col.max()) + 1)
            x[row, col] = 1.

            y = torch.empty(len(node_data), dtype=torch.long)
            for n_id, _, label in node_data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1]) as f:
            edge_data = f.read().split('\n')[1:-1]
            edge_indices = [[int(v) for v in r.split('\t')] for r in edge_data]
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
