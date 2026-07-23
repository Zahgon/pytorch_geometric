import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_tar


class OMDB(InMemoryDataset):

    url = 'https://omdb.mathub.io/dataset'

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
        from ase.io import read

        extract_tar(self.raw_paths[0], self.raw_dir, log=False)
        materials = read(osp.join(self.raw_dir, 'structures.xyz'), index=':')
        bandgaps = np.loadtxt(osp.join(self.raw_dir, 'bandgaps.csv'))

        data_list = []
        for material, bandgap in zip(materials, bandgaps):
            pos = torch.from_numpy(material.get_positions()).to(torch.float)
            z = torch.from_numpy(material.get_atomic_numbers()).to(torch.int64)
            y = torch.tensor([float(bandgap)])
            data_list.append(Data(z=z, pos=pos, y=y))

        train_data = data_list[:10000]
        test_data = data_list[10000:]

        if self.pre_filter is not None:
            train_data = [d for d in train_data if self.pre_filter(d)]
            test_data = [d for d in test_data if self.pre_filter(d)]

        if self.pre_transform is not None:
            train_data = [self.pre_transform(d) for d in train_data]
            test_data = [self.pre_transform(d) for d in test_data]

        self.save(train_data, self.processed_paths[0])
        self.save(test_data, self.processed_paths[1])
