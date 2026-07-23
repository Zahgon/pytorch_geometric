import copy
import os
import os.path as osp
import re
import sys
import warnings
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch.utils.data
from torch import Tensor

from torch_geometric.data.data import BaseData
from torch_geometric.io import fs

IndexType = Union[slice, Tensor, np.ndarray, Sequence]
MISSING = '???'


class Dataset(torch.utils.data.Dataset):
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        raise NotImplementedError

    def download(self) -> None:
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self) -> None:
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        raise NotImplementedError

    def get(self, idx: int) -> BaseData:
        r"""Gets the data object at index :obj:`idx`."""
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
        super().__init__()

        if isinstance(root, str):
            root = osp.expanduser(fs.normpath(root))

        self.root = root or MISSING
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log = log
        self._indices: Optional[Sequence] = None
        self.force_reload = force_reload

        if self.has_download:
            self._download()

        if self.has_process:
            self._process()

    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices

    @property
    def raw_dir(self) -> str:
        pass

    @property
    def processed_dir(self) -> str:
        pass

    @property
    def num_node_features(self) -> int:
        pass

    @property
    def num_features(self) -> int:
        pass

    @property
    def num_edge_features(self) -> int:
        pass

    def _infer_num_classes(self, y: Optional[Tensor]) -> int:
        pass

    @property
    def num_classes(self) -> int:
        pass

    @property
    def raw_paths(self) -> List[str]:
        pass

    @property
    def processed_paths(self) -> List[str]:
        pass

    @property
    def has_download(self) -> bool:
        pass

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        fs.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    @property
    def has_process(self) -> bool:
        pass

    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if not self.force_reload and osp.exists(f) and torch.load(
                f, weights_only=False) != _repr(self.pre_transform):
            warnings.warn(
                "The `pre_transform` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-processing technique, pass "
                "`force_reload=True` explicitly to reload the dataset.",
                stacklevel=2)

        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if not self.force_reload and osp.exists(f) and torch.load(
                f, weights_only=False) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, pass "
                "`force_reload=True` explicitly to reload the dataset.",
                stacklevel=2)

        if not self.force_reload and files_exist(self.processed_paths):
            return

        if self.log and 'PYTEST_CURRENT_TEST' not in os.environ:
            print('Processing...', file=sys.stderr)

        fs.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        fs.torch_save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        fs.torch_save(_repr(self.pre_filter), path)

        if self.log and 'PYTEST_CURRENT_TEST' not in os.environ:
            print('Done!', file=sys.stderr)

    def __len__(self) -> int:
        r"""The number of examples in the dataset."""
        return len(self.indices())

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union['Dataset', BaseData]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices.
        """
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            return self.index_select(idx)

    def __iter__(self) -> Iterator[BaseData]:
        for i in range(len(self)):
            yield self[i]

    def index_select(self, idx: IndexType) -> 'Dataset':
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        """
        indices = self.indices()

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def shuffle(
        self,
        return_perm: bool = False,
    ) -> Union['Dataset', Tuple['Dataset', Tensor]]:
        r"""Randomly shuffles the examples in the dataset.

        Args:
            return_perm (bool, optional): If set to :obj:`True`, will also
                return the random permutation used to shuffle the dataset.
                (default: :obj:`False`)
        """
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr})'

    def get_summary(self) -> Any:
        pass

    def print_summary(self, fmt: str = "psql") -> None:
        pass

    def to_datapipe(self) -> Any:
        pass


def overrides_method(cls, method_name: str) -> bool:
    pass


def to_list(value: Any) -> Sequence:
    pass


def files_exist(files: List[str]) -> bool:
    return len(files) != 0 and all([fs.exists(f) for f in files])


def _repr(obj: Any) -> str:
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', str(obj))


def _get_flattened_data_list(data_list: Iterable[Any]) -> List[BaseData]:
    pass
