import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce


class Airports(InMemoryDataset):
    edge_url = ('https://github.com/leoribeiro/struc2vec/'
                'raw/master/graph/{}-airports.edgelist')
    label_url = ('https://github.com/leoribeiro/struc2vec/'
                 'raw/master/graph/labels-{}-airports.txt')

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in ['usa', 'brazil', 'europe']
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
        download_url(self.edge_url.format(self.name), self.raw_dir)
        download_url(self.label_url.format(self.name), self.raw_dir)

    def process(self) -> None:
        index_map, ys = {}, []
        with open(self.raw_paths[1]) as f:
            rows = f.read().split('\n')[1:-1]
            for i, row in enumerate(rows):
                idx, label = row.split()
                index_map[int(idx)] = i
                ys.append(int(label))
        y = torch.tensor(ys, dtype=torch.long)
        x = torch.eye(y.size(0))

        edge_indices = []
        with open(self.raw_paths[0]) as f:
            rows = f.read().split('\n')[:-1]
            for row in rows:
                src, dst = row.split()
                edge_indices.append([index_map[int(src)], index_map[int(dst)]])
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_index = coalesce(edge_index, num_nodes=y.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Airports()'
