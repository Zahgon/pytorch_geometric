import os
import os.path as osp
from typing import Callable, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import read_txt_array


class PCPNetDataset(InMemoryDataset):

    url = 'http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip'

    category_files_train = {
        'NoNoise': 'trainingset_no_noise.txt',
        'Noisy': 'trainingset_whitenoise.txt',
        'VarDensity': 'trainingset_vardensity.txt',
        'NoisyAndVarDensity': 'trainingset_vardensity_whitenoise.txt'
    }

    category_files_val = {
        'NoNoise': 'validationset_no_noise.txt',
        'Noisy': 'validationset_whitenoise.txt',
        'VarDensity': 'validationset_vardensity.txt',
        'NoisyAndVarDensity': 'validationset_vardensity_whitenoise.txt'
    }

    category_files_test = {
        'All': 'testset_all.txt',
        'NoNoise': 'testset_no_noise.txt',
        'LowNoise': 'testset_low_noise.txt',
        'MedNoise': 'testset_med_noise.txt',
        'HighNoise': 'testset_high_noise.txt',
        'VarDensityStriped': 'testset_vardensity_striped.txt',
        'VarDensityGradient': 'testset_vardensity_gradient.txt'
    }

    def __init__(
        self,
        root: str,
        category: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:

        assert split in ['train', 'val', 'test']

        if split == 'train':
            assert category in self.category_files_train.keys()
        elif split == 'val':
            assert category in self.category_files_val.keys()
        else:
            assert category in self.category_files_test.keys()

        self.category = category
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        path_file = self.raw_paths
        with open(path_file[0]) as f:
            filenames = f.read().split('\n')[:-1]
        data_list = []
        for filename in filenames:
            pos_path = osp.join(self.raw_dir, filename + '.xyz')
            normal_path = osp.join(self.raw_dir, filename + '.normals')
            curv_path = osp.join(self.raw_dir, filename + '.curv')
            idx_path = osp.join(self.raw_dir, filename + '.pidx')
            pos = read_txt_array(pos_path)
            normals = read_txt_array(normal_path)
            curv = read_txt_array(curv_path)
            normals_and_curv = torch.cat([normals, curv], dim=1)
            test_idx = read_txt_array(idx_path, dtype=torch.long)
            data = Data(pos=pos, x=normals_and_curv)
            data.test_idx = test_idx
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'category={self.category})')
