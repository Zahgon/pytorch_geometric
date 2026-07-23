import os
import os.path as osp
from typing import Callable, List, Optional

import torch
from torch import Tensor

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class S3DIS(InMemoryDataset):

    url = ('https://shapenet.cs.stanford.edu/media/'
           'indoor3d_sem_seg_hdf5_data.zip')


    def __init__(
        self,
        root: str,
        test_area: int = 6,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert test_area >= 1 and test_area <= 6
        self.test_area = test_area
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
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        fs.rm(self.raw_dir)
        name = self.url.split('/')[-1].split('.')[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process(self) -> None:
        import h5py

        with open(self.raw_paths[0]) as f:
            filenames = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]

        with open(self.raw_paths[1]) as f:
            rooms = f.read().split('\n')[:-1]

        xs: List[Tensor] = []
        ys: List[Tensor] = []
        for filename in filenames:
            h5 = h5py.File(osp.join(self.raw_dir, filename))
            xs += torch.from_numpy(h5['data'][:]).unbind(0)
            ys += torch.from_numpy(h5['label'][:]).to(torch.long).unbind(0)

        test_area = f'Area_{self.test_area}'
        train_data_list, test_data_list = [], []
        for i, (x, y) in enumerate(zip(xs, ys)):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if test_area not in rooms[i]:
                train_data_list.append(data)
            else:
                test_data_list.append(data)

        self.save(train_data_list, self.processed_paths[0])
        self.save(test_data_list, self.processed_paths[1])
