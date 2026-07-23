import pickle
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class BA2MotifDataset(InMemoryDataset):
    url = 'https://github.com/flyingdoog/PGExplainer/raw/master/dataset'
    filename = 'BA-2motif.pkl'

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
    def raw_file_names(self) -> str:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        download_url(f'{self.url}/{self.filename}', self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0], 'rb') as f:
            adj, x, y = pickle.load(f)

        adjs = torch.from_numpy(adj)
        xs = torch.from_numpy(x).to(torch.float)
        ys = torch.from_numpy(y)

        data_list: List[Data] = []
        for i in range(xs.size(0)):
            edge_index = adjs[i].nonzero().t()
            x = xs[i]
            y = int(ys[i].nonzero())

            data = Data(x=x, edge_index=edge_index, y=y)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
