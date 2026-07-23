from typing import Any, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkLoader, NodeLoader
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.utils import infer_filter_per_worker


class ZipLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        loaders: Union[List[NodeLoader], List[LinkLoader]],
        filter_per_worker: Optional[bool] = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(loaders[0].data)

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        for loader in loaders:
            if not callable(getattr(loader, 'collate_fn', None)):
                raise ValueError("'{loader.__class__.__name__}' does not have "
                                 "a 'collate_fn' method")
            if not callable(getattr(loader, 'filter_fn', None)):
                raise ValueError("'{loader.__class__.__name__}' does not have "
                                 "a 'filter_fn' method")
            loader.filter_per_worker = filter_per_worker

        iterator = range(min([len(loader.dataset) for loader in loaders]))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

        self.loaders = loaders
        self.filter_per_worker = filter_per_worker

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Tuple[Data, ...], Tuple[HeteroData, ...]]:
        r"""Samples subgraphs from a batch of input IDs."""
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def collate_fn(self, index: List[int]) -> Tuple[Any, ...]:
        pass

    def filter_fn(
        self,
        outs: Tuple[Any, ...],
    ) -> Tuple[Union[Data, HeteroData], ...]:
        pass

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()

        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(loaders={self.loaders})'
