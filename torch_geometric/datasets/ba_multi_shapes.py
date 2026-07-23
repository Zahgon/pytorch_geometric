import pickle
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class BAMultiShapesDataset(InMemoryDataset):
    url = ('https://github.com/steveazzolin/gnn_logic_global_expl/raw/master/'
           'datasets/BAMultiShapes/BAMultiShapes.pkl')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0], 'rb') as f:
            adjs, xs, ys = pickle.load(f)

        data_list: List[Data] = []
        for adj, x, y in zip(adjs, xs, ys):
            edge_index = torch.from_numpy(adj).nonzero().t()
            x = torch.from_numpy(np.array(x)).to(torch.float)

            data = Data(x=x, edge_index=edge_index, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
