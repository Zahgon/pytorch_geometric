import copy
import os.path as osp
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
from torch import Tensor
from tqdm import tqdm

import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.data.separate import separate
from torch_geometric.io import fs


class InMemoryDataset(Dataset):
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        raise NotImplementedError

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, log,
                         force_reload)

        self._data: Optional[BaseData] = None
        self.slices: Optional[Dict[str, Tensor]] = None
        self._data_list: Optional[MutableSequence[Optional[BaseData]]] = None

    @property
    def num_classes(self) -> int:
        pass

    def len(self) -> int:
        pass

    def get(self, idx: int) -> BaseData:
        if self.len() == 1:
            return copy.copy(self._data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)

        return data

    @classmethod
    def save(cls, data_list: Sequence[BaseData], path: str) -> None:
        r"""Saves a list of data objects to the file path :obj:`path`."""
        data, slices = cls.collate(data_list)
        fs.torch_save((data.to_dict(), slices, data.__class__), path)

    def load(self, path: str, data_cls: Type[BaseData] = Data) -> None:
        r"""Loads the dataset from the file path :obj:`path`."""
        out = fs.torch_load(path)
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    @staticmethod
    def collate(
        data_list: Sequence[BaseData],
    ) -> Tuple[BaseData, Optional[Dict[str, Tensor]]]:
        r"""Collates a list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects to the internal
        storage format of :class:`~torch_geometric.data.InMemoryDataset`.
        """
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    def copy(self, idx: Optional[IndexType] = None) -> 'InMemoryDataset':
        r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,
        will clone the full dataset. Otherwise, will only clone a subset of the
        dataset from indices :obj:`idx`.
        Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or
        :obj:`np.ndarray` of type long or bool.
        """
        if idx is None:
            data_list = [self.get(i) for i in self.indices()]
        else:
            data_list = [self.get(i) for i in self.index_select(idx).indices()]

        dataset = copy.copy(self)
        dataset._indices = None
        dataset._data_list = None
        dataset.data, dataset.slices = self.collate(data_list)
        return dataset

    def to_on_disk_dataset(
        self,
        root: Optional[str] = None,
        backend: str = 'sqlite',
        log: bool = True,
    ) -> 'torch_geometric.data.OnDiskDataset':
        pass

    @property
    def data(self) -> Any:
        pass

    @data.setter
    def data(self, value: Any):
        pass

    def __getattr__(self, key: str) -> Any:
        data = self.__dict__.get('_data')
        if isinstance(data, Data) and key in data:
            if self._indices is None and data.__inc__(key, data[key]) == 0:
                return data[key]
            else:
                data_list = [self.get(i) for i in self.indices()]
                return Batch.from_data_list(data_list)[key]

        raise AttributeError(f"'{self.__class__.__name__}' object has no "
                             f"attribute '{key}'")

    def to(self, device: Union[int, str]) -> 'InMemoryDataset':
        r"""Performs device conversion of the whole dataset."""
        if self._indices is not None:
            raise ValueError("The given 'InMemoryDataset' only references a "
                             "subset of examples of the full dataset")
        if self._data_list is not None:
            raise ValueError("The data of the dataset is already cached")
        self._data.to(device)
        return self

    def cpu(self, *args: str) -> 'InMemoryDataset':
        r"""Moves the dataset to CPU memory."""
        return self.to(torch.device('cpu'))

    def cuda(
        self,
        device: Optional[Union[int, str]] = None,
    ) -> 'InMemoryDataset':
        pass


def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
    pass
