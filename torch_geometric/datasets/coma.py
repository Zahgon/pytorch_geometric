import os.path as osp
from glob import glob
from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.io import read_ply


class CoMA(InMemoryDataset):

    url = 'https://coma.is.tue.mpg.de/'

    categories = [
        'bareteeth',
        'cheeks_in',
        'eyebrow',
        'high_smile',
        'lips_back',
        'lips_up',
        'mouth_down',
        'mouth_extreme',
        'mouth_middle',
        'mouth_open',
        'mouth_side',
        'mouth_up',
    ]

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
            f"Dataset not found. Please download 'COMA_data.zip' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    def process(self) -> None:
        folders = sorted(glob(osp.join(self.raw_dir, 'FaceTalk_*')))
        if len(folders) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            folders = sorted(glob(osp.join(self.raw_dir, 'FaceTalk_*')))

        train_data_list, test_data_list = [], []
        for folder in folders:
            for i, category in enumerate(self.categories):
                files = sorted(glob(osp.join(folder, category, '*.ply')))
                for j, f in enumerate(files):
                    data = read_ply(f)
                    data.y = torch.tensor([i], dtype=torch.long)
                    if self.pre_filter is not None and\
                       not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if (j % 100) < 90:
                        train_data_list.append(data)
                    else:
                        test_data_list.append(data)

        self.save(train_data_list, self.processed_paths[0])
        self.save(test_data_list, self.processed_paths[1])
