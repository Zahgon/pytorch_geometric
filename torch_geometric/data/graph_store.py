import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor

from torch_geometric.index import index2ptr, ptr2index
from torch_geometric.typing import EdgeTensorType, EdgeType, OptTensor
from torch_geometric.utils import index_sort
from torch_geometric.utils.mixin import CastMixin

ConversionOutputType = Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor],
                             Dict[EdgeType, OptTensor]]


class EdgeLayout(Enum):
    COO = 'coo'
    CSC = 'csc'
    CSR = 'csr'


@dataclass
class EdgeAttr(CastMixin):

    edge_type: EdgeType

    layout: EdgeLayout

    is_sorted: bool = False

    size: Optional[Tuple[int, int]] = None

    def __init__(
        self,
        edge_type: EdgeType,
        layout: EdgeLayout,
        is_sorted: bool = False,
        size: Optional[Tuple[int, int]] = None,
    ):
        layout = EdgeLayout(layout)

        if layout == EdgeLayout.CSR and is_sorted:
            raise ValueError("Cannot create a 'CSR' edge attribute with "
                             "option 'is_sorted=True'")

        if layout == EdgeLayout.CSC:
            is_sorted = True

        self.edge_type = edge_type
        self.layout = layout
        self.is_sorted = is_sorted
        self.size = size


class GraphStore(ABC):
    def __init__(self, edge_attr_cls: Optional[Any] = None):
        super().__init__()
        self.__dict__['_edge_attr_cls'] = edge_attr_cls or EdgeAttr


    @abstractmethod
    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        r"""To be implemented by :class:`GraphStore` subclasses."""

    def put_edge_index(self, edge_index: EdgeTensorType, *args,
                       **kwargs) -> bool:
        r"""Synchronously adds an :obj:`edge_index` tuple to the
        :class:`GraphStore`.
        Returns whether insertion was successful.

        Args:
            edge_index (Tuple[torch.Tensor, torch.Tensor]): The
                :obj:`edge_index` tuple in a format specified in
                :class:`EdgeAttr`.
            *args: Arguments passed to :class:`EdgeAttr`.
            **kwargs: Keyword arguments passed to :class:`EdgeAttr`.
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._put_edge_index(edge_index, edge_attr)

    @abstractmethod
    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        r"""To be implemented by :class:`GraphStore` subclasses."""

    def get_edge_index(self, *args, **kwargs) -> EdgeTensorType:
        r"""Synchronously obtains an :obj:`edge_index` tuple from the
        :class:`GraphStore`.

        Args:
            *args: Arguments passed to :class:`EdgeAttr`.
            **kwargs: Keyword arguments passed to :class:`EdgeAttr`.

        Raises:
            KeyError: If the :obj:`edge_index` corresponding to the input
                :class:`EdgeAttr` was not found.
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        edge_index = self._get_edge_index(edge_attr)
        if edge_index is None:
            raise KeyError(f"'edge_index' for '{edge_attr}' not found")
        return edge_index

    @abstractmethod
    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        r"""To be implemented by :class:`GraphStore` subclasses."""

    def remove_edge_index(self, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        r"""Returns all registered edge attributes."""


    def coo(
        self,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:
        r"""Returns the edge indices in the :class:`GraphStore` in COO format.

        Args:
            edge_types (List[Any], optional): The edge types of edge indices
                to obtain. If set to :obj:`None`, will return the edge indices
                of all existing edge types. (default: :obj:`None`)
            store (bool, optional): Whether to store converted edge indices in
                the :class:`GraphStore`. (default: :obj:`False`)
        """
        return self._edges_to_layout(EdgeLayout.COO, edge_types, store)

    def csr(
        self,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:
        r"""Returns the edge indices in the :class:`GraphStore` in CSR format.

        Args:
            edge_types (List[Any], optional): The edge types of edge indices
                to obtain. If set to :obj:`None`, will return the edge indices
                of all existing edge types. (default: :obj:`None`)
            store (bool, optional): Whether to store converted edge indices in
                the :class:`GraphStore`. (default: :obj:`False`)
        """
        return self._edges_to_layout(EdgeLayout.CSR, edge_types, store)

    def csc(
        self,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:
        r"""Returns the edge indices in the :class:`GraphStore` in CSC format.

        Args:
            edge_types (List[Any], optional): The edge types of edge indices
                to obtain. If set to :obj:`None`, will return the edge indices
                of all existing edge types. (default: :obj:`None`)
            store (bool, optional): Whether to store converted edge indices in
                the :class:`GraphStore`. (default: :obj:`False`)
        """
        return self._edges_to_layout(EdgeLayout.CSC, edge_types, store)


    def __setitem__(self, key: EdgeAttr, value: EdgeTensorType):
        self.put_edge_index(value, key)

    def __getitem__(self, key: EdgeAttr) -> Optional[EdgeTensorType]:
        return self.get_edge_index(key)

    def __delitem__(self, key: EdgeAttr):
        return self.remove_edge_index(key)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


    def _edge_to_layout(
        self,
        attr: EdgeAttr,
        layout: EdgeLayout,
        store: bool = False,
    ) -> Tuple[Tensor, Tensor, OptTensor]:

        (row, col), perm = self.get_edge_index(attr), None

        if layout == EdgeLayout.COO:  # COO output requested:
            if attr.layout == EdgeLayout.CSR:  # CSR->COO
                row = ptr2index(row)
            elif attr.layout == EdgeLayout.CSC:  # CSC->COO
                col = ptr2index(col)

        elif layout == EdgeLayout.CSR:  # CSR output requested:
            if attr.layout == EdgeLayout.CSC:  # CSC->COO
                col = ptr2index(col)

            if attr.layout != EdgeLayout.CSR:  # COO->CSR
                num_rows = attr.size[0] if attr.size is not None else int(
                    row.max()) + 1
                row, perm = index_sort(row, max_value=num_rows)
                col = col[perm]
                row = index2ptr(row, num_rows)

        else:  # CSC output requested:
            if attr.layout == EdgeLayout.CSR:  # CSR->COO
                row = ptr2index(row)

            if attr.layout != EdgeLayout.CSC:  # COO->CSC
                if hasattr(self, 'meta') and self.meta.get('is_hetero', False):
                    num_cols = int(col.max()) + 1
                elif attr.size is not None:
                    num_cols = attr.size[1]
                else:
                    num_cols = int(col.max()) + 1

                if not attr.is_sorted:  # Not sorted by destination.
                    col, perm = index_sort(col, max_value=num_cols)
                    row = row[perm]
                col = index2ptr(col, num_cols)

        if attr.layout != layout and store:
            attr = copy.copy(attr)
            attr.layout = layout
            if perm is not None:
                attr.is_sorted = False
            self.put_edge_index((row, col), attr)

        return row, col, perm

    def _edges_to_layout(
        self,
        layout: EdgeLayout,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:

        edge_attrs: List[EdgeAttr] = self.get_all_edge_attrs()

        if hasattr(self, 'meta'):  # `LocalGraphStore` hack.
            is_hetero = self.meta.get('is_hetero', False)
        else:
            is_hetero = all(attr.edge_type is not None for attr in edge_attrs)

        if not is_hetero:
            return self._edge_to_layout(edge_attrs[0], layout, store)

        edge_type_attrs: Dict[EdgeType, List[EdgeAttr]] = defaultdict(list)
        for attr in self.get_all_edge_attrs():
            edge_type_attrs[attr.edge_type].append(attr)

        if edge_types is not None:
            for edge_type in edge_types:
                if edge_type not in edge_type_attrs:
                    raise ValueError(f"The 'edge_index' of type '{edge_type}' "
                                     f"was not found in the graph store.")

            edge_type_attrs = {
                key: attr
                for key, attr in edge_type_attrs.items() if key in edge_types
            }

        row_dict, col_dict, perm_dict = {}, {}, {}
        for edge_type, attrs in edge_type_attrs.items():
            layouts = [attr.layout for attr in attrs]

            if layout in layouts:  # No conversion needed.
                attr = attrs[layouts.index(layout)]
            elif EdgeLayout.COO in layouts:  # Prefer COO for conversion.
                attr = attrs[layouts.index(EdgeLayout.COO)]
            elif EdgeLayout.CSC in layouts:
                attr = attrs[layouts.index(EdgeLayout.CSC)]
            elif EdgeLayout.CSR in layouts:
                attr = attrs[layouts.index(EdgeLayout.CSR)]

            row_dict[edge_type], col_dict[edge_type], perm_dict[edge_type] = (
                self._edge_to_layout(attr, layout, store))

        return row_dict, col_dict, perm_dict
