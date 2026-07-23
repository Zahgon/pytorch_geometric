import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.utils import index_to_mask


class DGraphFin(InMemoryDataset):

    url = "https://dgraph.xinye.com"

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

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self) -> str:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    @property
    def num_classes(self) -> int:
        pass

    def process(self) -> None:
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        path = osp.join(self.raw_dir, "dgraphfin.npz")

        with np.load(path) as loader:
            x = torch.from_numpy(loader['x']).to(torch.float)
            y = torch.from_numpy(loader['y']).to(torch.long)
            edge_index = torch.from_numpy(loader['edge_index']).to(torch.long)
            edge_type = torch.from_numpy(loader['edge_type']).to(torch.long)
            edge_time = torch.from_numpy(loader['edge_timestamp']).to(
                torch.long)
            train_nodes = torch.from_numpy(loader['train_mask']).to(torch.long)
            val_nodes = torch.from_numpy(loader['valid_mask']).to(torch.long)
            test_nodes = torch.from_numpy(loader['test_mask']).to(torch.long)

            train_mask = index_to_mask(train_nodes, size=x.size(0))
            val_mask = index_to_mask(val_nodes, size=x.size(0))
            test_mask = index_to_mask(test_nodes, size=x.size(0))
            data = Data(x=x, edge_index=edge_index.t(), edge_type=edge_type,
                        edge_time=edge_time, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
