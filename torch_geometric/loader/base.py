from typing import Any, Callable

from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
)


class DataLoaderIterator:
    def __init__(self, iterator: _BaseDataLoaderIter, transform_fn: Callable):
        self.iterator = iterator
        self.transform_fn = transform_fn

    def __iter__(self) -> 'DataLoaderIterator':
        return self

    def _reset(self, loader: Any, first_iter: bool = False):
        self.iterator._reset(loader, first_iter)

    def __len__(self) -> int:
        return len(self.iterator)

    def __next__(self) -> Any:
        return self.transform_fn(next(self.iterator))

    def __del__(self) -> Any:
        if isinstance(self.iterator, _MultiProcessingDataLoaderIter):
            self.iterator.__del__()
