from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader


def to_device(inputs: Any, device: Optional[torch.device] = None) -> Any:
    pass


class CachedLoader:
    def __init__(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
        transform: Optional[Callable] = None,
    ):
        self.loader = loader
        self.device = device
        self.transform = transform

        self._cache: List[Any] = []

    def clear(self):
        r"""Clears the cache."""
        self._cache = []

    def __iter__(self) -> Any:
        if len(self._cache):
            for batch in self._cache:
                yield batch
            return

        for batch in self.loader:

            if self.transform is not None:
                batch = self.transform(batch)

            batch = to_device(batch, self.device)

            self._cache.append(batch)

            yield batch

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'
