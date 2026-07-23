import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class GDELTLite(InMemoryDataset):
    url = 'https://data.pyg.org/datasets/gdelt_lite.zip'

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
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        import pandas as pd

        x = fs.torch_load(self.raw_paths[0])
        df = pd.read_csv(self.raw_paths[1])
        edge_attr = fs.torch_load(self.raw_paths[2])

        row = torch.from_numpy(df['src'].values)
        col = torch.from_numpy(df['dst'].values)
        edge_index = torch.stack([row, col], dim=0)
        time = torch.from_numpy(df['time'].values).to(torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, time=time)
        data = data if self.pre_transform is None else self.pre_transform(data)

        self.save([data], self.processed_paths[0])
