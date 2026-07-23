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
from torch_geometric.io import read_txt_array


class TOSCA(InMemoryDataset):

    url = 'http://tosca.cs.technion.ac.il/data/toscahires-asci.zip'

    categories = [
        'cat', 'centaur', 'david', 'dog', 'gorilla', 'horse', 'michael',
        'victoria', 'wolf'
    ]

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        categories = self.categories if categories is None else categories
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.categories = categories
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
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        data_list = []
        for cat in self.categories:
            paths = glob.glob(osp.join(self.raw_dir, f'{cat}*.tri'))
            paths = [path[:-4] for path in paths]
            paths = sorted(paths, key=lambda e: (len(e), e))

            for path in paths:
                pos = read_txt_array(f'{path}.vert')
                face = read_txt_array(f'{path}.tri', dtype=torch.long)
                face = face - face.min()  # Ensure zero-based index.
                data = Data(pos=pos, face=face.t().contiguous())
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        self.save(data_list, self.processed_paths[0])
