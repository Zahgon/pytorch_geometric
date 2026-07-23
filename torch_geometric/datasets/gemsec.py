import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class GemsecDeezer(InMemoryDataset):

    url = 'https://graphmining.ai/datasets/ptg/gemsec'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        assert self.name in ['HU', 'HR', 'RO']
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
    def raw_file_names(self) -> str:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        download_url(osp.join(self.url, self.name + '.npz'), self.raw_dir)

    def process(self) -> None:
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        y = torch.from_numpy(data['target']).to(torch.long)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
