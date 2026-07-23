import glob
import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs, read_off


class ModelNet(InMemoryDataset):

    urls = {
        '10':
        'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',  # noqa
        '40': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    }

    def __init__(
        self,
        root: str,
        name: str = '10',
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert name in ['10', '40']
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = osp.join(self.root, f'ModelNet{self.name}')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

        metadata_folder = osp.join(self.root, '__MACOSX')
        if osp.exists(metadata_folder):
            fs.rm(metadata_folder)

    def process(self) -> None:
        self.save(self.process_set('train'), self.processed_paths[0])
        self.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset: str) -> List[Data]:
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob(f'{folder}/{category}_*.off')
            for path in paths:
                data = read_off(path)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return data_list

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name}({len(self)})'
