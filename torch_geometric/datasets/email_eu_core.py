import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)


class EmailEUCore(InMemoryDataset):

    urls = [
        'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
        'https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz'
    ]

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
        for url in self.urls:
            path = download_url(url, self.raw_dir)
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        import pandas as pd

        edge_index = pd.read_csv(self.raw_paths[0], sep=' ', header=None)
        edge_index = torch.from_numpy(edge_index.values).t().contiguous()

        y = pd.read_csv(self.raw_paths[1], sep=' ', header=None, usecols=[1])
        y = torch.from_numpy(y.values).view(-1)

        data = Data(edge_index=edge_index, y=y, num_nodes=y.size(0))

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
