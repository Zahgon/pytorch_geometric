from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class LastFMAsia(InMemoryDataset):

    url = 'https://graphmining.ai/datasets/ptg/lastfm_asia.npz'

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
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = torch.from_numpy(data['features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
