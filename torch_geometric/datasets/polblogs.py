import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)


class PolBlogs(InMemoryDataset):

    url = 'https://netset.telecom-paris.fr/datasets/polblogs.tar.gz'

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
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        import pandas as pd

        edge_index = pd.read_csv(self.raw_paths[0], header=None, sep='\t',
                                 usecols=[0, 1])
        edge_index = torch.from_numpy(edge_index.values).t().contiguous()

        y = pd.read_csv(self.raw_paths[1], header=None, sep='\t')
        y = torch.from_numpy(y.values).view(-1)

        data = Data(edge_index=edge_index, y=y, num_nodes=y.size(0))

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
