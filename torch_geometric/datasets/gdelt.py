from typing import Callable, List, Optional

import torch
from torch import Tensor

from torch_geometric.data import download_url
from torch_geometric.datasets.icews import EventDataset
from torch_geometric.io import read_txt_array


class GDELT(EventDataset):

    url = 'https://github.com/INK-USC/RE-Net/raw/master/data/GDELT'
    splits = [0, 1734399, 1973164, 2278405]  # Train/Val/Test splits.

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        idx = self.processed_file_names.index(f'{split}.pt')
        self.load(self.processed_paths[idx])

    @property
    def num_nodes(self) -> int:
        return 7691

    @property
    def num_rels(self) -> int:
        pass

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process_events(self) -> Tensor:
        events = []
        for path in self.raw_paths:
            data = read_txt_array(path, sep='\t', end=4, dtype=torch.long)
            data[:, 3] = data[:, 3] // 15
            events += [data]
        return torch.cat(events, dim=0)

    def process(self) -> None:
        s = self.splits
        data_list = self._process_data_list()
        self.save(data_list[s[0]:s[1]], self.processed_paths[0])
        self.save(data_list[s[1]:s[2]], self.processed_paths[1])
        self.save(data_list[s[2]:s[3]], self.processed_paths[2])
