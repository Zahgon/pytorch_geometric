import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class BrcaTcga(InMemoryDataset):
    url = 'https://zenodo.org/record/8251328/files/brca_tcga.zip?download=1'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        fs.rm(self.raw_dir)
        os.rename(osp.join(self.root, 'brca_tcga'), self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        graph_feat = pd.read_csv(self.raw_paths[0], index_col=0).values
        graph_feat = torch.from_numpy(graph_feat).to(torch.float)
        graph_labels = np.loadtxt(self.raw_paths[1], delimiter=',')
        graph_label = torch.from_numpy(graph_labels).to(torch.float)
        edge_index = fs.torch_load(self.raw_paths[2])

        data_list = []
        for x, y in zip(graph_feat, graph_label):
            data = Data(x=x.view(-1, 1), edge_index=edge_index, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
