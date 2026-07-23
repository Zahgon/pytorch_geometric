import functools
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
import torch.utils._pytree as pytree
import xxhash
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import CPUHashMap, CUDAHashMap

aten = torch.ops.aten

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}


def implements(torch_function: Callable) -> Callable:
    r"""Registers a :pytorch:`PyTorch` function override."""
    @functools.wraps(torch_function)
    def decorator(my_function: Callable) -> Callable:
        pass

    return decorator


def as_key_tensor(
    key: Any,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    pass


def get_hash_map(key: Tensor) -> Union[CPUHashMap, CUDAHashMap]:
    pass


class HashTensor(Tensor):
    _map: Union[Tensor, CPUHashMap, CUDAHashMap]
    _value: Optional[Tensor]
    _min_key: Tensor
    _max_key: Tensor

    @staticmethod
    def __new__(
        cls: Type,
        key: Any,
        value: Optional[Any] = None,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> 'HashTensor':

        if value is not None:
            value = torch.as_tensor(value, dtype=dtype, device=device)
            device = value.device

        key = as_key_tensor(key, device=device)

        if key.dim() != 1:
            raise ValueError(f"'key' data in '{cls.__name__}' needs to be "
                             f"one-dimensional (got {key.dim()} dimensions)")

        if not key.is_contiguous():
            raise ValueError(f"'key' data in '{cls.__name__}' needs to be "
                             f"contiguous")

        if value is not None:
            if key.device != value.device:
                raise ValueError(f"'key' and 'value' data in '{cls.__name__}' "
                                 f"are expected to be on the same device (got "
                                 f"'{key.device}' and '{value.device}')")

            if key.numel() != value.size(0):
                raise ValueError(f"'key' and 'value' data in '{cls.__name__}' "
                                 f"are expected to have the same size in the "
                                 f"first dimension (got {key.size(0)} and "
                                 f"{value.size(0)})")

        min_key = key.min() if key.numel() > 0 else key.new_zeros(())
        max_key = key.max() if key.numel() > 0 else key.new_zeros(())

        _range = max_key - min_key
        if (key.dtype in {torch.uint8, torch.int16} or _range <= 1_000_000
                or _range <= 2 * key.numel()):
            _map = torch.full(
                size=(_range + 3, ),
                fill_value=-1,
                dtype=torch.int64,
                device=key.device,
            )
            _map[key.long() - (min_key.long() - 1)] = torch.arange(
                key.numel(),
                dtype=_map.dtype,
                device=_map.device,
            )
        else:
            _map = get_hash_map(key)

        return cls._from_data(
            _map,
            value,
            min_key,
            max_key,
            num_keys=key.numel(),
            dtype=dtype,
        )


    @classmethod
    def _from_data(
        cls,
        _map: Union[Tensor, CPUHashMap, CUDAHashMap],
        value: Optional[Tensor],
        min_key: Tensor,
        max_key: Tensor,
        *,
        num_keys: int,
        dtype: Optional[torch.dtype],
    ) -> 'HashTensor':

        if value is not None:
            dtype = value.dtype
            size = value.size()
            stride = value.stride()
            layout = value.layout
            requires_grad = value.requires_grad
        else:
            dtype = dtype or torch.int64
            size = torch.Size([num_keys])
            stride = (1, )
            layout = torch.strided
            requires_grad = False

        out = Tensor._make_wrapper_subclass(
            cls,
            size=size,
            strides=stride,
            dtype=dtype,
            device=min_key.device,
            layout=layout,
            requires_grad=requires_grad,
        )
        assert isinstance(out, HashTensor)

        out._map = _map
        out._value = value
        out._min_key = min_key
        out._max_key = max_key

        return out

    @property
    def _key(self) -> Tensor:
        pass

    def _shallow_copy(self) -> 'HashTensor':
        pass

    def _get(self, query: Tensor) -> Tensor:
        if isinstance(self._map, Tensor):
            index = query.long() - (self._min_key.long() - 1)
            index = self._map[index.clamp_(min=0, max=self._map.numel() - 1)]
        elif torch_geometric.typing.WITH_CUDA_HASH_MAP and query.is_cuda:
            index = self._map.get(query)
        elif torch_geometric.typing.WITH_CPU_HASH_MAP:
            index = self._map.get(query.cpu())
        else:
            import pandas as pd

            ser = pd.Series(query.cpu().numpy(), dtype=self._map)
            index = torch.from_numpy(ser.cat.codes.to_numpy().copy()).long()

        index = index.to(self.device)

        if self._value is None:
            return index.to(self.dtype)

        out = self._value[index]

        mask = index != -1
        mask = mask.view([-1] + [1] * (out.dim() - 1))
        fill_value = float('NaN') if out.is_floating_point() else -1
        if torch_geometric.typing.WITH_PT20:
            other: Union[int, float, Tensor] = fill_value
        else:
            other = torch.full_like(out, fill_value)

        return out.where(mask, other)


    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`HashTensor` representation back to a
        :class:`torch.Tensor` representation.
        """
        if self._value is not None:
            return self._value
        return torch.arange(self.size(0), dtype=self.dtype, device=self.device)


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

        args = pytree.tree_map_only(HashTensor, lambda x: x.as_tensor(), args)
        if kwargs is not None:
            kwargs = pytree.tree_map_only(HashTensor, lambda x: x.as_tensor(),
                                          kwargs)
        return func(*args, **(kwargs or {}))

    def __tensor_flatten__(self) -> Tuple[List[str], Tuple[Any, ...]]:
        attrs = ['_map', '_min_key', '_max_key']
        if self._value is not None:
            attrs.append('_value')

        ctx = (self.size(0), self.dtype)

        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict[str, Any],
        ctx: Tuple[Any, ...],
        outer_size: Tuple[int, ...],
        outer_stride: Tuple[int, ...],
    ) -> 'HashTensor':
        return HashTensor._from_data(
            inner_tensors['_map'],
            inner_tensors.get('_value', None),
            inner_tensors['_min_key'],
            inner_tensors['_min_key'],
            num_keys=ctx[0],
            dtype=ctx[1],
        )

    def __repr__(self) -> str:  # type: ignore
        indent = len(f'{self.__class__.__name__}(')
        tensor_str = torch._tensor_str._tensor_str(self.as_tensor(), indent)
        return torch._tensor_str._str_intern(self, tensor_contents=tensor_str)

    def tolist(self) -> List[Any]:
        """"""  # noqa: D419
        return self.as_tensor().tolist()

    def numpy(self, *, force: bool = False) -> np.ndarray:
        """"""  # noqa: D419
        return self.as_tensor().numpy(force=force)

    def index_select(  # type: ignore
        self,
        dim: int,
        index: Any,
    ) -> Union['HashTensor', Tensor]:
        """"""  # noqa: D419
        return torch.index_select(self, dim, index)

    def select(  # type: ignore
        self,
        dim: int,
        index: Any,
    ) -> Union['HashTensor', Tensor]:
        """"""  # noqa: D419
        return torch.select(self, dim, index)

    def share_memory_(self) -> 'HashTensor':
        """"""  # noqa: D419
        if isinstance(self._map, Tensor):
            self._map.share_memory_()
        if self._value is not None:
            self._value.share_memory_()
        self._min_key.share_memory_()
        self._max_key.share_memory_()
        return self

    def is_shared(self) -> bool:
        pass

    def detach_(self) -> 'HashTensor':
        """"""  # noqa: D419
        if self._value is not None:
            self._value.detach_()
        return super().detach_()  # type: ignore

    def __getitem__(self, indices: Any) -> Union['HashTensor', Tensor]:
        if not isinstance(indices, tuple):
            indices = (indices, )
        assert len(indices) > 0

        if indices[0] is Ellipsis and len(indices) > 1:
            nonempty_indices = [i for i in indices[1:] if i is not None]
            if len(nonempty_indices) == self.dim():
                indices = indices[1:]

        if isinstance(indices[0], (int, bool)):
            index: Union[int, Tensor] = int(as_key_tensor([indices[0]]))
            indices = (index, ) + indices[1:]
        elif isinstance(indices[0], (Tensor, list, np.ndarray)):
            index = as_key_tensor(indices[0], device=self.device)
            indices = (index, ) + indices[1:]

        indices = indices[0] if len(indices) == 1 else indices

        return super().__getitem__(indices)


@implements(aten.alias.default)
def _alias(tensor: HashTensor) -> HashTensor:
    pass


@implements(aten.clone.default)
def _clone(
    tensor: HashTensor,
    *,
    memory_format: torch.memory_format = torch.preserve_format,
) -> HashTensor:
    pass


@implements(aten.detach.default)
def _detach(tensor: HashTensor) -> HashTensor:
    pass


@implements(aten._to_copy.default)
def _to_copy(
    tensor: HashTensor,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> HashTensor:
    pass


@implements(aten._pin_memory.default)
def _pin_memory(tensor: HashTensor) -> HashTensor:
    pass


@implements(aten.unsqueeze.default)
def _unsqueeze(tensor: HashTensor, dim: int) -> HashTensor:
    pass


@implements(aten.squeeze.default)
def _squeeze_default(tensor: HashTensor) -> HashTensor:
    pass


@implements(aten.squeeze.dim)
@implements(getattr(aten.squeeze, 'dims', aten.squeeze.dim))
def _squeeze_dim(
    tensor: HashTensor,
    dim: Union[int, List[int]],
) -> HashTensor:
    pass


@implements(aten.slice.Tensor)
def _slice(
    tensor: HashTensor,
    dim: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> HashTensor:
    pass


_old_index_select = torch.index_select


def _new_index_select(
    input: Tensor,
    dim: Union[int, str],
    index: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    pass


torch.index_select = _new_index_select  # type: ignore


@implements(aten.index_select.default)
def _index_select(
    tensor: HashTensor,
    dim: int,
    index: Tensor,
) -> Union[HashTensor, Tensor]:

    if dim == 0 or dim == -tensor.dim():
        return tensor._get(index)

    return tensor._from_data(
        tensor._map,
        aten.index_select.default(tensor.as_tensor(), dim, index),
        tensor._min_key,
        tensor._max_key,
        num_keys=tensor.size(0),
        dtype=tensor.dtype,
    )


_old_select = torch.select


def _new_select(
    input: Tensor,
    dim: Union[int, str],
    index: int,
) -> Tensor:
    pass


torch.select = _new_select  # type: ignore


@implements(aten.select.int)
def _select(
    tensor: HashTensor,
    dim: int,
    index: int,
) -> Union[HashTensor, Tensor]:
    pass


@implements(aten.index.Tensor)
def _index(
    tensor: HashTensor,
    indices: List[Optional[Tensor]],
) -> Union[HashTensor, Tensor]:
    pass
