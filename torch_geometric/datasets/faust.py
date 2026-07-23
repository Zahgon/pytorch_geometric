import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.io import fs, read_ply


class FAUST(InMemoryDataset):

    url = 'http://faust.is.tue.mpg.de/'

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)

    @property
    def raw_file_names(self) -> str:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    def process(self) -> None:
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        path = osp.join(self.raw_dir, 'MPI-FAUST', 'training', 'registrations')
        path = osp.join(path, 'tr_reg_{0:03d}.ply')
        data_list = []
        for i in range(100):
            data = read_ply(path.format(i))
            data.y = torch.tensor([i % 10], dtype=torch.long)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        self.save(data_list[:80], self.processed_paths[0])
        self.save(data_list[80:], self.processed_paths[1])

        fs.rm(osp.join(self.raw_dir, 'MPI-FAUST'))
