import copy
import warnings
from typing import Any, List, Optional

import torch
from torch import Tensor
from typing_extensions import Self

from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import select
from torch_geometric.utils._subgraph import hyper_subgraph


class HyperGraphData(Data):
    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            **kwargs,
        )

    @property
    def num_edges(self) -> int:
        pass

    @property
    def num_nodes(self) -> Optional[int]:
        num_nodes = super().num_nodes

        if (self.edge_index is not None and num_nodes == self.num_edges):
            return max(self.edge_index[0]) + 1

        return num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: Optional[int]) -> None:
        self._store.num_nodes = num_nodes

    def is_edge_attr(self, key: str) -> bool:
        val = super().is_edge_attr(key)
        if not val and self.edge_index is not None:
            return key in self and self[key].size(0) == self.num_edges
        return val

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == 'edge_index':
            return torch.tensor([[self.num_nodes], [self.num_edges]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def subgraph(self, subset: Tensor) -> 'HyperGraphData':
        r"""Returns the induced subgraph given by the node indices
        :obj:`subset`.

        .. note::

            If only a subset of a hyperedge's nodes are to be
            selected in the subgraph, the hyperedge will remain in the
            subgraph, but only the selected nodes will be connected by
            the hyperedge. Hyperedges that only connects one node in the
            subgraph will be removed.

        Examples:
            >>> x = torch.randn(4, 16)
            >>> edge_index = torch.tensor([
            ...     [0, 1, 0, 2, 1, 1, 2, 4],
            ...     [0, 0, 1, 1, 1, 2, 2, 2]
            >>> ])
            >>> data = HyperGraphData(x = x, edge_index = edge_index)
            >>> subset = torch.tensor([1, 2, 4])
            >>> subgraph = data.subgraph(subset)
            >>> subgraph.edge_index
            tensor([[2, 1, 1, 2, 4],
            [0, 0, 1, 1, 1]])

        Args:
            subset (LongTensor or BoolTensor): The nodes to keep.
        """
        assert self.edge_index is not None
        out = hyper_subgraph(subset, self.edge_index, relabel_nodes=True,
                             num_nodes=self.num_nodes, return_edge_mask=True)
        edge_index, _, edge_mask = out

        data = copy.copy(self)

        for key, value in self.items():
            if key == 'edge_index':
                data.edge_index = edge_index
            elif key == 'num_nodes':
                if subset.dtype == torch.bool:
                    data.num_nodes = int(subset.sum())
                else:
                    data.num_nodes = subset.size(0)
            elif self.is_node_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                data[key] = select(value, subset, dim=cat_dim)
            elif self.is_edge_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                data[key] = select(value, edge_mask, dim=cat_dim)

        return data

    def edge_subgraph(self, subset: Tensor) -> Self:
        raise NotImplementedError

    def to_heterogeneous(
        self,
        node_type: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        node_type_names: Optional[List[NodeType]] = None,
        edge_type_names: Optional[List[EdgeType]] = None,
    ) -> HeteroData:
        raise NotImplementedError

    def has_isolated_nodes(self) -> bool:
        if self.edge_index is None:
            return False
        return torch.unique(self.edge_index[0]).size(0) < self.num_nodes

    def is_directed(self) -> bool:
        raise NotImplementedError

    def is_undirected(self) -> bool:
        raise NotImplementedError

    def has_self_loops(self) -> bool:
        raise NotImplementedError

    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the data."""
        cls_name = self.__class__.__name__
        status = True

        num_nodes = self.num_nodes
        if num_nodes is None:
            status = False
            warn_or_raise(f"'num_nodes' is undefined in '{cls_name}'",
                          raise_on_error)

        if self.edge_index is not None:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                status = False
                warn_or_raise(
                    f"'edge_index' needs to be of shape [2, num_edges] in "
                    f"'{cls_name}' (found {self.edge_index.size()})",
                    raise_on_error)

        if self.edge_index is not None and self.edge_index.numel() > 0:
            if self.edge_index.min() < 0:
                status = False
                warn_or_raise(
                    f"'edge_index' contains negative indices in "
                    f"'{cls_name}' (found {int(self.edge_index.min())})",
                    raise_on_error)

            if num_nodes is not None and self.edge_index[0].max() >= num_nodes:
                status = False
                warn_or_raise(
                    f"'edge_index' contains larger indices than the number "
                    f"of nodes ({num_nodes}) in '{cls_name}' "
                    f"(found {int(self.edge_index.max())})", raise_on_error)

        return status


def warn_or_raise(msg: str, raise_on_error: bool = True) -> None:
    if raise_on_error:
        raise ValueError(msg)
    else:
        warnings.warn(msg, stacklevel=2)
