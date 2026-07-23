import os.path as osp
from typing import Callable, Optional

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_npz


class CitationFull(InMemoryDataset):

    url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/{}.npz'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        to_undirected: bool = True,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        self.to_undirected = to_undirected
        assert self.name in ['cora', 'cora_ml', 'citeseer', 'dblp', 'pubmed']
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        pass

    @property
    def processed_dir(self) -> str:
        pass

    @property
    def raw_file_names(self) -> str:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self) -> None:
        data = read_npz(self.raw_paths[0], to_undirected=self.to_undirected)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Full()'


class CoraFull(CitationFull):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, 'cora', transform, pre_transform)

    def download(self) -> None:
        super().download()

    def process(self) -> None:
        super().process()
