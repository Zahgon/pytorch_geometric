import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.typing import FeatureTensorType, NodeType
from torch_geometric.utils.mixin import CastMixin

IndexType = Union[torch.Tensor, np.ndarray, slice, int]


class _FieldStatus(Enum):
    UNSET = None


@dataclass
class TensorAttr(CastMixin):

    group_name: Optional[NodeType] = _FieldStatus.UNSET

    attr_name: Optional[str] = _FieldStatus.UNSET

    index: Optional[IndexType] = _FieldStatus.UNSET


    def is_set(self, key: str) -> bool:
        r"""Whether an attribute is set in :obj:`TensorAttr`."""
        assert key in self.__dataclass_fields__
        return getattr(self, key) != _FieldStatus.UNSET

    def is_fully_specified(self) -> bool:
        r"""Whether the :obj:`TensorAttr` has no unset fields."""
        return all([self.is_set(key) for key in self.__dataclass_fields__])

    def update(self, attr: 'TensorAttr') -> 'TensorAttr':
        r"""Updates an :class:`TensorAttr` with set attributes from another
        :class:`TensorAttr`.
        """
        for key in self.__dataclass_fields__:
            if attr.is_set(key):
                setattr(self, key, getattr(attr, key))
        return self


class AttrView(CastMixin):
    def __init__(self, store: 'FeatureStore', attr: TensorAttr):
        self.__dict__['_store'] = store
        self.__dict__['_attr'] = attr


    def __getattr__(self, key: Any) -> Union['AttrView', FeatureTensorType]:
        r"""Sets the first unset field of the backing :class:`TensorAttr`
        object to the attribute.

        This allows for :class:`AttrView` to be indexed by different values of
        attributes, in order.
        In particular, for a feature store that we want to index by
        :obj:`group_name` and :obj:`attr_name`, the following code will do so:

        .. code-block:: python

            store[group, attr]
            store[group].attr
            store.group.attr
        """
        out = copy.copy(self)

        attr_name: Optional[str] = None
        for field in out._attr.__dataclass_fields__:
            if getattr(out._attr, field) == _FieldStatus.UNSET:
                attr_name = field
                break

        if attr_name is None:
            raise AttributeError(f"Cannot access attribute '{key}' on view "
                                 f"'{out}' as all attributes have already "
                                 f"been set in this view")

        setattr(out._attr, attr_name, key)

        if out._attr.is_fully_specified():
            return out._store.get_tensor(out._attr)

        return out

    def __getitem__(self, key: Any) -> Union['AttrView', FeatureTensorType]:
        r"""Sets the first unset field of the backing :class:`TensorAttr`
        object to the attribute via indexing.

        This allows for :class:`AttrView` to be indexed by different values of
        attributes, in order.
        In particular, for a feature store that we want to index by
        :obj:`group_name` and :obj:`attr_name`, the following code will do so:

        .. code-block:: python

            store[group, attr]
            store[group][attr]

        """
        return self.__getattr__(key)


    def __setattr__(self, key: str, value: Any):
        r"""Supports attribute assignment to the backing :class:`TensorAttr` of
        an :class:`AttrView`.

        This allows for :class:`AttrView` objects to set their backing
        attribute values.
        In particular, the following operation sets the :obj:`index` of an
        :class:`AttrView`:

        .. code-block:: python

            view = store.view(group_name)
            view.index = torch.tensor([1, 2, 3])
        """
        if key not in self._attr.__dataclass_fields__:
            raise ValueError(f"Attempted to set nonexistent attribute '{key}' "
                             f"(acceptable attributes are "
                             f"{self._attr.__dataclass_fields__})")

        setattr(self._attr, key, value)

    def __setitem__(self, key: str, value: Any):
        r"""Supports attribute assignment to the backing :class:`TensorAttr` of
        an :class:`AttrView` via indexing.

        This allows for :class:`AttrView` objects to set their backing
        attribute values.
        In particular, the following operation sets the `index` of an
        :class:`AttrView`:

        .. code-block:: python

            view = store.view(TensorAttr(group_name))
            view['index'] = torch.tensor([1, 2, 3])
        """
        self.__setattr__(key, value)


    def __call__(self) -> FeatureTensorType:
        r"""Supports :class:`AttrView` as a callable to force retrieval from
        the currently specified attributes.

        In particular, this passes the current :class:`TensorAttr` object to a
        GET call, regardless of whether all attributes have been specified.
        It returns the result of this call.
        In particular, the following operation returns a tensor by performing a
        GET operation on the backing feature store:

        .. code-block:: python

            store[group_name, attr_name]()
        """
        attr = copy.copy(self._attr)
        for key in attr.__dataclass_fields__:  # Set all UNSET values to None.
            if not attr.is_set(key):
                setattr(attr, key, None)
        return self._store.get_tensor(attr)

    def __copy__(self) -> 'AttrView':
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_attr'] = copy.copy(out.__dict__['_attr'])
        return out

    def __eq__(self, obj: Any) -> bool:
        r"""Compares two :class:`AttrView` objects by checking equality of
        their :class:`FeatureStore` references and :class:`TensorAttr`
        attributes.
        """
        if not isinstance(obj, AttrView):
            return False
        return self._store == obj._store and self._attr == obj._attr

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(store={self._store}, '
                f'attr={self._attr})')




class FeatureStore(ABC):
    _tensor_attr_cls: TensorAttr

    def __init__(self, tensor_attr_cls: Optional[Any] = None):
        super().__init__()
        self.__dict__['_tensor_attr_cls'] = tensor_attr_cls or TensorAttr


    @abstractmethod
    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        r"""To be implemented by :class:`FeatureStore` subclasses."""

    def put_tensor(self, tensor: FeatureTensorType, *args, **kwargs) -> bool:
        r"""Synchronously adds a :obj:`tensor` to the :class:`FeatureStore`.
        Returns whether insertion was successful.

        Args:
            tensor (torch.Tensor or np.ndarray): The feature tensor to be
                added.
            *args: Arguments passed to :class:`TensorAttr`.
            **kwargs: Keyword arguments passed to :class:`TensorAttr`.

        Raises:
            ValueError: If the input :class:`TensorAttr` is not fully
                specified.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_fully_specified():
            raise ValueError(f"The input TensorAttr '{attr}' is not fully "
                             f"specified. Please fully-specify the input by "
                             f"specifying all 'UNSET' fields")
        return self._put_tensor(tensor, attr)

    @abstractmethod
    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        r"""To be implemented by :class:`FeatureStore` subclasses."""

    def get_tensor(
        self,
        *args,
        convert_type: bool = False,
        **kwargs,
    ) -> FeatureTensorType:
        r"""Synchronously obtains a :class:`tensor` from the
        :class:`FeatureStore`.

        Args:
            *args: Arguments passed to :class:`TensorAttr`.
            convert_type (bool, optional): Whether to convert the type of the
                output tensor to the type of the attribute index.
                (default: :obj:`False`)
            **kwargs: Keyword arguments passed to :class:`TensorAttr`.

        Raises:
            ValueError: If the input :class:`TensorAttr` is not fully
                specified.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_fully_specified():
            raise ValueError(f"The input TensorAttr '{attr}' is not fully "
                             f"specified. Please fully-specify the input by "
                             f"specifying all 'UNSET' fields.")

        tensor = self._get_tensor(attr)
        if convert_type:
            tensor = self._to_type(attr, tensor)
        return tensor

    def _multi_get_tensor(
        self,
        attrs: List[TensorAttr],
    ) -> List[Optional[FeatureTensorType]]:
        r"""To be implemented by :class:`FeatureStore` subclasses."""
        return [self._get_tensor(attr) for attr in attrs]

    def multi_get_tensor(
        self,
        attrs: List[TensorAttr],
        convert_type: bool = False,
    ) -> List[FeatureTensorType]:
        r"""Synchronously obtains a list of tensors from the
        :class:`FeatureStore` for each tensor associated with the attributes in
        :obj:`attrs`.

        .. note::
            The default implementation simply iterates over all calls to
            :meth:`get_tensor`. Implementer classes that can provide
            additional, more performant functionality are recommended to
            to override this method.

        Args:
            attrs (List[TensorAttr]): A list of input :class:`TensorAttr`
                objects that identify the tensors to obtain.
            convert_type (bool, optional): Whether to convert the type of the
                output tensor to the type of the attribute index.
                (default: :obj:`False`)

        Raises:
            ValueError: If any input :class:`TensorAttr` is not fully
                specified.
        """
        attrs = [self._tensor_attr_cls.cast(attr) for attr in attrs]
        bad_attrs = [attr for attr in attrs if not attr.is_fully_specified()]
        if len(bad_attrs) > 0:
            raise ValueError(
                f"The input TensorAttr(s) '{bad_attrs}' are not fully "
                f"specified. Please fully-specify them by specifying all "
                f"'UNSET' fields")

        tensors = self._multi_get_tensor(attrs)
        if convert_type:
            tensors = [
                self._to_type(attr, tensor)
                for attr, tensor in zip(attrs, tensors)
            ]
        return tensors

    @abstractmethod
    def _remove_tensor(self, attr: TensorAttr) -> bool:
        r"""To be implemented by :obj:`FeatureStore` subclasses."""

    def remove_tensor(self, *args, **kwargs) -> bool:
        pass

    def update_tensor(self, tensor: FeatureTensorType, *args,
                      **kwargs) -> bool:
        pass


    @abstractmethod
    def _get_tensor_size(self, attr: TensorAttr) -> Optional[Tuple[int, ...]]:
        pass

    def get_tensor_size(self, *args, **kwargs) -> Optional[Tuple[int, ...]]:
        r"""Obtains the size of a tensor given its :class:`TensorAttr`, or
        :obj:`None` if the tensor does not exist.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_set('index'):
            attr.index = None
        return self._get_tensor_size(attr)

    @abstractmethod
    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        r"""Returns all registered tensor attributes."""


    def view(self, *args, **kwargs) -> AttrView:
        r"""Returns a view of the :class:`FeatureStore` given a not yet
        fully-specified :class:`TensorAttr`.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        return AttrView(self, attr)


    @staticmethod
    def _to_type(
        attr: TensorAttr,
        tensor: FeatureTensorType,
    ) -> FeatureTensorType:
        if isinstance(attr.index, Tensor) and isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor)
        if isinstance(attr.index, np.ndarray) and isinstance(tensor, Tensor):
            return tensor.detach().cpu().numpy()
        return tensor


    def __setitem__(self, key: TensorAttr, value: FeatureTensorType):
        r"""Supports :obj:`store[tensor_attr] = tensor`."""
        key = self._tensor_attr_cls.cast(key)
        assert key.is_fully_specified()
        self.put_tensor(value, key)

    def __getitem__(self, key: TensorAttr) -> Any:
        r"""Supports pythonic indexing into the :class:`FeatureStore`.

        In particular, the following rules are followed for indexing:

        * A fully-specified :obj:`key` will produce a tensor output.

        * A partially-specified :obj:`key` will produce an :class:`AttrView`
          output, which is a view on the :class:`FeatureStore`. If a view is
          called, it will produce a tensor output from the corresponding
          (partially specified) attributes.
        """
        attr = self._tensor_attr_cls.cast(key)
        if attr.is_fully_specified():
            return self.get_tensor(attr)
        return self.view(attr)

    def __delitem__(self, attr: TensorAttr):
        r"""Supports :obj:`del store[tensor_attr]`."""
        attr = self._tensor_attr_cls.cast(attr)
        attr = copy.copy(attr)
        for key in attr.__dataclass_fields__:  # Set all UNSET values to None.
            if not attr.is_set(key):
                setattr(attr, key, None)
        self.remove_tensor(attr)

    def __iter__(self):
        raise NotImplementedError

    def __eq__(self, obj: object) -> bool:
        return id(self) == id(obj)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
