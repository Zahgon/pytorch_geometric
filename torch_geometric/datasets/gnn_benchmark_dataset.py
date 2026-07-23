import logging
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
from torch_geometric.utils import remove_self_loops


class GNNBenchmarkDataset(InMemoryDataset):

    names = ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']

    root_url = 'https://data.pyg.org/datasets/benchmarking-gnns'
    urls = {
        'PATTERN': f'{root_url}/PATTERN_v2.zip',
        'CLUSTER': f'{root_url}/CLUSTER_v2.zip',
        'MNIST': f'{root_url}/MNIST_v2.zip',
        'CIFAR10': f'{root_url}/CIFAR10_v2.zip',
        'TSP': f'{root_url}/TSP_v2.zip',
        'CSL': 'https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1',
    }

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        assert self.name in self.names

        if self.name == 'CSL' and split != 'train':
            split = 'train'
            logging.warning(
                "Dataset 'CSL' does not provide a standardized splitting. "
                "Instead, it is recommended to perform 5-fold cross "
                "validation with stratifed sampling")

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"Split '{split}' found, but expected either "
                             f"'train', 'val', or 'test'")

        self.load(path)

    @property
    def raw_dir(self) -> str:
        pass

    @property
    def processed_dir(self) -> str:
        pass

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        path = download_url(self.urls[self.name], self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        if self.name == 'CSL':
            data_list = self.process_CSL()
            self.save(data_list, self.processed_paths[0])
        else:
            inputs = fs.torch_load(self.raw_paths[0])
            for i in range(len(inputs)):
                data_list = [Data(**data_dict) for data_dict in inputs[i]]

                if self.pre_filter is not None:
                    data_list = [d for d in data_list if self.pre_filter(d)]

                if self.pre_transform is not None:
                    data_list = [self.pre_transform(d) for d in data_list]

                self.save(data_list, self.processed_paths[i])

    def process_CSL(self) -> List[Data]:
        with open(self.raw_paths[0], 'rb') as f:
            adjs = pickle.load(f)

        ys = fs.torch_load(self.raw_paths[1]).tolist()

        data_list = []
        for adj, y in zip(adjs, ys):
            row, col = torch.from_numpy(adj.row), torch.from_numpy(adj.col)
            edge_index = torch.stack([row, col], dim=0).to(torch.long)
            edge_index, _ = remove_self_loops(edge_index)
            data = Data(edge_index=edge_index, y=y, num_nodes=adj.shape[0])
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        return data_list

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
