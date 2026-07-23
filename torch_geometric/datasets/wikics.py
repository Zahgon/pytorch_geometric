import json
import warnings
from itertools import chain
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_undirected


class WikiCS(InMemoryDataset):

    url = 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        is_undirected: Optional[bool] = None,
        force_reload: bool = False,
    ) -> None:
        if is_undirected is None:
            warnings.warn(
                f"The {self.__class__.__name__} dataset now returns an "
                f"undirected graph by default. Please explicitly specify "
                f"'is_undirected=False' to restore the old behavior.",
                stacklevel=2)
            is_undirected = True
        self.is_undirected = is_undirected
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def download(self) -> None:
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0]) as f:
            data = json.load(f)

        x = torch.tensor(data['features'], dtype=torch.float)
        y = torch.tensor(data['labels'], dtype=torch.long)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))  # type: ignore
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        if self.is_undirected:
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))

        train_mask = torch.tensor(data['train_masks'], dtype=torch.bool)
        train_mask = train_mask.t().contiguous()

        val_mask = torch.tensor(data['val_masks'], dtype=torch.bool)
        val_mask = val_mask.t().contiguous()

        test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)

        stopping_mask = torch.tensor(data['stopping_masks'], dtype=torch.bool)
        stopping_mask = stopping_mask.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask,
                    stopping_mask=stopping_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
