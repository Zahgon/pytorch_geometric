import functools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch import Tensor

from torch_geometric.typing import INDEX_DTYPES

aten = torch.ops.aten

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}


def ptr2index(ptr: Tensor, output_size: Optional[int] = None) -> Tensor:
    index = torch.arange(ptr.numel() - 1, dtype=ptr.dtype, device=ptr.device)
    return index.repeat_interleave(ptr.diff(), output_size=output_size)


def index2ptr(index: Tensor, size: Optional[int] = None) -> Tensor:
    if size is None:
        size = int(index.max()) + 1 if index.numel() > 0 else 0

    return torch._convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype != torch.int64)


class CatMetadata(NamedTuple):
    nnz: List[int]
    dim_size: List[Optional[int]]
    is_sorted: List[bool]


def implements(torch_function: Callable) -> Callable:
    r"""Registers a :pytorch:`PyTorch` function override."""
    @functools.wraps(torch_function)
    def decorator(my_function: Callable) -> Callable:
        pass

    return decorator


def assert_valid_dtype(tensor: Tensor) -> None:
    if tensor.dtype not in INDEX_DTYPES:
        raise ValueError(f"'Index' holds an unsupported data type "
                         f"(got '{tensor.dtype}', but expected one of "
                         f"{INDEX_DTYPES})")


def assert_one_dimensional(tensor: Tensor) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"'Index' needs to be one-dimensional "
                         f"(got {tensor.dim()} dimensions)")


def assert_contiguous(tensor: Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError("'Index' needs to be contiguous. Please call "
                         "`index.contiguous()` before proceeding.")


def assert_sorted(func: Callable) -> Callable:
    pass


class Index(Tensor):

    _data: Tensor

    _dim_size: Optional[int] = None

    _is_sorted: bool = False

    _indptr: Optional[Tensor] = None

    _cat_metadata: Optional[CatMetadata] = None

    @staticmethod
    def __new__(
        cls: Type,
        data: Any,
        *args: Any,
        dim_size: Optional[int] = None,
        is_sorted: bool = False,
        **kwargs: Any,
    ) -> 'Index':
        if not isinstance(data, Tensor):
            data = torch.tensor(data, *args, **kwargs)
        elif len(args) > 0:
            raise TypeError(
                f"new() received an invalid combination of arguments - got "
                f"(Tensor, {', '.join(str(type(arg)) for arg in args)})")
        elif len(kwargs) > 0:
            raise TypeError(f"new() received invalid keyword arguments - got "
                            f"{set(kwargs.keys())})")

        assert isinstance(data, Tensor)

        indptr: Optional[Tensor] = None

        if isinstance(data, cls):  # If passed `Index`, inherit metadata:
            indptr = data._indptr
            dim_size = dim_size or data.dim_size
            is_sorted = is_sorted or data.is_sorted

        assert_valid_dtype(data)
        assert_one_dimensional(data)
        assert_contiguous(data)

        out = Tensor._make_wrapper_subclass(
            cls,
            size=data.size(),
            strides=data.stride(),
            dtype=data.dtype,
            device=data.device,
            layout=data.layout,
            requires_grad=False,
        )
        assert isinstance(out, Index)

        out._data = data
        out._dim_size = dim_size
        out._is_sorted = is_sorted
        out._indptr = indptr

        if isinstance(data, cls):
            out._data = data._data

            if dim_size is not None and dim_size != data.dim_size:
                out._indptr = None

        return out


    def validate(self) -> 'Index':
        r"""Validates the :class:`Index` representation.

        In particular, it ensures that

        * it only holds valid indices.
        * the sort order is correctly set.
        """
        assert_valid_dtype(self._data)
        assert_one_dimensional(self._data)
        assert_contiguous(self._data)

        if self.numel() > 0 and self._data.min() < 0:
            raise ValueError(f"'{self.__class__.__name__}' contains negative "
                             f"indices (got {int(self.min())})")

        if (self.numel() > 0 and self.dim_size is not None
                and self._data.max() >= self.dim_size):
            raise ValueError(f"'{self.__class__.__name__}' contains larger "
                             f"indices than its registered size "
                             f"(got {int(self._data.max())}, but expected "
                             f"values smaller than {self.dim_size})")

        if self.is_sorted and (self._data.diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted")

        return self


    @property
    def dim_size(self) -> Optional[int]:
        pass

    @property
    def is_sorted(self) -> bool:
        pass

    @property
    def dtype(self) -> torch.dtype:  # type: ignore
        return self._data.dtype


    def get_dim_size(self) -> int:
        r"""The size of the underlying sparse vector.
        Automatically computed and cached when not explicitly set.
        """
        if self._dim_size is None:
            dim_size = int(self._data.max()) + 1 if self.numel() > 0 else 0
            self._dim_size = dim_size

        assert isinstance(self._dim_size, int)
        return self._dim_size

    def dim_resize_(self, dim_size: Optional[int]) -> 'Index':
        pass

    @assert_sorted
    def get_indptr(self) -> Tensor:
        r"""Returns the compressed index representation in case :class:`Index`
        is sorted.
        """
        if self._indptr is None:
            self._indptr = index2ptr(self._data, self.get_dim_size())

        assert isinstance(self._indptr, Tensor)
        return self._indptr

    def fill_cache_(self) -> 'Index':
        pass


    def share_memory_(self) -> 'Index':
        """"""  # noqa: D419
        self._data.share_memory_()
        if self._indptr is not None:
            self._indptr.share_memory_()
        return self

    def is_shared(self) -> bool:
        pass

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`Index` representation back to a
        :class:`torch.Tensor` representation.
        """
        return self._data


    def __tensor_flatten__(self) -> Tuple[List[str], Tuple[Any, ...]]:
        attrs = ['_data']
        if self._indptr is not None:
            attrs.append('_indptr')

        ctx = (
            self._dim_size,
            self._is_sorted,
            self._cat_metadata,
        )

        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict[str, Any],
        ctx: Tuple[Any, ...],
        outer_size: Tuple[int, ...],
        outer_stride: Tuple[int, ...],
    ) -> 'Index':
        index = Index(
            inner_tensors['_data'],
            dim_size=ctx[0],
            is_sorted=ctx[1],
        )

        index._indptr = inner_tensors.get('_indptr', None)
        index._cat_metadata = ctx[2]

        return index

    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls: Type,
        func: Callable[..., Any],
        types: Iterable[Type[Any]],
        args: Iterable[Tuple[Any, ...]] = (),
        kwargs: Optional[Dict[Any, Any]] = None,
    ) -> Any:

        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        args = pytree.tree_map_only(Index, lambda x: x._data, args)
        if kwargs is not None:
            kwargs = pytree.tree_map_only(Index, lambda x: x._data, kwargs)
        return func(*args, **(kwargs or {}))

    def __repr__(self) -> str:  # type: ignore
        prefix = f'{self.__class__.__name__}('
        indent = len(prefix)
        tensor_str = torch._tensor_str._tensor_str(self._data, indent)

        suffixes = []
        if self.dim_size is not None:
            suffixes.append(f'dim_size={self.dim_size}')
        if (self.device.type != torch._C._get_default_device()
                or (self.device.type == 'cuda'
                    and torch.cuda.current_device() != self.device.index)
                or (self.device.type == 'mps')):
            suffixes.append(f"device='{self.device}'")
        if self.dtype != torch.int64:
            suffixes.append(f'dtype={self.dtype}')
        if self.is_sorted:
            suffixes.append('is_sorted=True')

        return torch._tensor_str._add_suffixes(prefix + tensor_str, suffixes,
                                               indent, force_newline=False)

    def tolist(self) -> List[Any]:
        """"""  # noqa: D419
        return self._data.tolist()

    def numpy(self, *, force: bool = False) -> np.ndarray:
        """"""  # noqa: D419
        return self._data.numpy(force=force)


    def _shallow_copy(self) -> 'Index':
        pass

    def _clear_metadata(self) -> 'Index':
        self._dim_size = None
        self._is_sorted = False
        self._indptr = None
        self._cat_metadata = None
        return self


def apply_(
    tensor: Index,
    fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> Union[Index, Tensor]:

    data = fn(tensor._data, *args, **kwargs)

    if data.dtype not in INDEX_DTYPES:
        return data

    if tensor._data.data_ptr() != data.data_ptr():
        out = Index(data)
    else:  # In-place:
        tensor._data = data
        out = tensor

    out._dim_size = tensor._dim_size
    out._is_sorted = tensor._is_sorted
    out._cat_metadata = tensor._cat_metadata

    if tensor._indptr is not None:
        out._indptr = fn(tensor._indptr, *args, **kwargs)

    return out


@implements(aten.clone.default)
def _clone(
    tensor: Index,
    *,
    memory_format: torch.memory_format = torch.preserve_format,
) -> Index:
    pass


@implements(aten._to_copy.default)
def _to_copy(
    tensor: Index,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Union[Index, Tensor]:
    pass


@implements(aten.alias.default)
def _alias(tensor: Index) -> Index:
    pass


@implements(aten._pin_memory.default)
def _pin_memory(tensor: Index) -> Index:
    pass


@implements(aten.sort.default)
def _sort(
    tensor: Index,
    dim: int = -1,
    descending: bool = False,
) -> Tuple[Index, Tensor]:
    pass


@implements(aten.sort.stable)
def _sort_stable(
    tensor: Index,
    *,
    stable: bool = False,
    dim: int = -1,
    descending: bool = False,
) -> Tuple[Index, Tensor]:
    pass


@implements(aten.cat.default)
def _cat(
    tensors: List[Union[Index, Tensor]],
    dim: int = 0,
) -> Union[Index, Tensor]:

    data_list = pytree.tree_map_only(Index, lambda x: x._data, tensors)
    data = aten.cat.default(data_list, dim=dim)

    if any([not isinstance(tensor, Index) for tensor in tensors]):
        return data

    out = Index(data)

    nnz_list = [t.numel() for t in tensors]
    dim_size_list = [t.dim_size for t in tensors]  # type: ignore
    is_sorted_list = [t.is_sorted for t in tensors]  # type: ignore

    total_dim_size: Optional[int] = 0
    for dim_size in dim_size_list:
        if dim_size is None:
            total_dim_size = None
            break
        assert isinstance(total_dim_size, int)
        total_dim_size = max(dim_size, total_dim_size)

    out._dim_size = total_dim_size

    out._cat_metadata = CatMetadata(
        nnz=nnz_list,
        dim_size=dim_size_list,
        is_sorted=is_sorted_list,
    )

    return out


@implements(aten.flip.default)
def _flip(
    input: Index,
    dims: Union[List[int], Tuple[int, ...]],
) -> Index:
    pass


@implements(aten.index_select.default)
def _index_select(
    input: Union[Index, Tensor],
    dim: int,
    index: Union[Index, Tensor],
) -> Union[Index, Tensor]:

    out = aten.index_select.default(
        input._data if isinstance(input, Index) else input,
        dim,
        index._data if isinstance(index, Index) else index,
    )

    if isinstance(input, Index):
        out = Index(out)
        out._dim_size = input.dim_size

    return out


@implements(aten.slice.Tensor)
def _slice(
    input: Index,
    dim: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> Index:
    pass


@implements(aten.index.Tensor)
def _index(
    input: Union[Index, Tensor],
    indices: List[Optional[Union[Tensor, Index]]],
) -> Union[Index, Tensor]:
    pass


@implements(aten.add.Tensor)
def _add(
    input: Union[int, Tensor, Index],
    other: Union[int, Tensor, Index],
    *,
    alpha: int = 1,
) -> Union[Index, Tensor]:
    pass


@implements(aten.add_.Tensor)
def add_(
    input: Index,
    other: Union[int, Tensor, Index],
    *,
    alpha: int = 1,
) -> Index:

    dim_size = input.dim_size
    is_sorted = input.is_sorted
    input._clear_metadata()

    aten.add_.Tensor(
        input._data,
        other._data if isinstance(other, Index) else other,
        alpha=alpha,
    )

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        if dim_size is not None:
            input._dim_size = dim_size + alpha * other
        input._is_sorted = is_sorted

    elif isinstance(other, Index):
        if dim_size is not None and other.dim_size is not None:
            input._dim_size = dim_size + alpha * other.dim_size

    return input


@implements(aten.sub.Tensor)
def _sub(
    input: Union[int, Tensor, Index],
    other: Union[int, Tensor, Index],
    *,
    alpha: int = 1,
) -> Union[Index, Tensor]:
    pass


@implements(aten.sub_.Tensor)
def sub_(
    input: Index,
    other: Union[int, Tensor, Index],
    *,
    alpha: int = 1,
) -> Index:

    dim_size = input.dim_size
    is_sorted = input.is_sorted
    input._clear_metadata()

    aten.sub_.Tensor(
        input._data,
        other._data if isinstance(other, Index) else other,
        alpha=alpha,
    )

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        if dim_size is not None:
            input._dim_size = dim_size - alpha * other
        input._is_sorted = is_sorted

    return input
