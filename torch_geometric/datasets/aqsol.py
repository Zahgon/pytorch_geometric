import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class AQSOL(InMemoryDataset):
    url = 'https://www.dropbox.com/s/lzu9lmukwov12kt/aqsol_graph_raw.zip?dl=1'

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        fs.rm(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'asqol_graph_raw'), self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graphs = pickle.load(f)

            data_list: List[Data] = []
            for graph in graphs:
                x, edge_attr, edge_index, y = graph

                x = torch.from_numpy(x)
                edge_attr = torch.from_numpy(edge_attr)
                edge_index = torch.from_numpy(edge_index)
                y = torch.tensor([y]).float()

                if edge_index.numel() == 0:
                    continue  # Skipping for graphs with no bonds/edges.

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            self.save(data_list, path)

    def atoms(self) -> List[str]:
        pass

    def bonds(self) -> List[str]:
        pass
