import json
import os
from typing import Callable, List, Optional

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class AirfRANS(InMemoryDataset):
    url = 'https://data.isir.upmc.fr/extrality/pytorch_geometric/AirfRANS.zip'
    tasks = ['full', 'scarce', 'reynolds', 'aoa']

    def __init__(
        self,
        root: str,
        task: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if task not in self.tasks:
            raise ValueError(f"Expected 'task' to be in {self.tasks} "
                             f"got '{task}'")

        self.task = 'full' if task == 'scarce' and not train else task
        self.split = 'train' if train else 'test'

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
        with open(self.raw_paths[1]) as f:
            manifest = json.load(f)
        total = manifest['full_train'] + manifest['full_test']
        partial = set(manifest[f'{self.task}_{self.split}'])

        data_list = []
        raw_data = fs.torch_load(self.raw_paths[0])
        for k, s in enumerate(total):
            if s in partial:
                data = Data(**raw_data[k])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'task={self.task}, split={self.split})')
