import os
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

from torch import Tensor

from torch_geometric.data import Database, RocksDatabase, SQLiteDatabase
from torch_geometric.data.data import BaseData
from torch_geometric.data.database import Schema
from torch_geometric.data.dataset import Dataset


class OnDiskDataset(Dataset):
    BACKENDS = {
        'sqlite': SQLiteDatabase,
        'rocksdb': RocksDatabase,
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        backend: str = 'sqlite',
        schema: Schema = object,
        log: bool = True,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"Database backend must be one of "
                             f"{set(self.BACKENDS.keys())} "
                             f"(got '{backend}')")

        self.backend = backend
        self.schema = schema

        self._db: Optional[Database] = None
        self._numel: Optional[int] = None

        super().__init__(root, transform, pre_filter=pre_filter, log=log)

    @property
    def processed_file_names(self) -> str:
        pass

    @property
    def db(self) -> Database:
        pass

    def close(self) -> None:
        r"""Closes the connection to the underlying database."""
        if self._db is not None:
            self._db.close()

    def serialize(self, data: BaseData) -> Any:
        r"""Serializes the :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object into the expected DB
        schema.
        """
        if self.schema == object:
            return data
        raise NotImplementedError(f"`{self.__class__.__name__}.serialize()` "
                                  f"needs to be overridden in case a "
                                  f"non-default schema was passed")

    def deserialize(self, data: Any) -> BaseData:
        r"""Deserializes the DB entry into a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object.
        """
        if self.schema == object:
            return data
        raise NotImplementedError(f"`{self.__class__.__name__}.deserialize()` "
                                  f"needs to be overridden in case a "
                                  f"non-default schema was passed")

    def append(self, data: BaseData) -> None:
        r"""Appends the data object to the dataset."""
        index = len(self)
        self.db.insert(index, self.serialize(data))
        self._numel += 1

    def extend(
        self,
        data_list: Sequence[BaseData],
        batch_size: Optional[int] = None,
    ) -> None:
        r"""Extends the dataset by a list of data objects."""
        start = len(self)
        end = start + len(data_list)
        data_list = [self.serialize(data) for data in data_list]
        self.db.multi_insert(range(start, end), data_list, batch_size)
        self._numel += (end - start)

    def get(self, idx: int) -> BaseData:
        r"""Gets the data object at index :obj:`idx`."""
        return self.deserialize(self.db.get(idx))

    def multi_get(
        self,
        indices: Union[Iterable[int], Tensor, slice, range],
        batch_size: Optional[int] = None,
    ) -> List[BaseData]:
        pass

    def __getitems__(self, indices: List[int]) -> List[BaseData]:
        return self.multi_get(indices)

    def len(self) -> int:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'
