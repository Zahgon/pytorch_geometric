import copy
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_torch_csc_tensor


class RootedSubgraphData(Data):
    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == 'sub_edge_index':
            return self.n_id.size(0)
        if key in ['n_sub_batch', 'e_sub_batch']:
            return 1 + int(self.n_sub_batch[-1])
        elif key == 'n_id':
            return self.num_nodes
        elif key == 'e_id':
            assert self.edge_index is not None
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

    def map_data(self) -> Data:
        pass


class RootedSubgraph(BaseTransform, ABC):
    @abstractmethod
    def extract(
        self,
        data: Data,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        pass

    def map(
        self,
        data: Data,
        n_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        pass

    def forward(self, data: Data) -> RootedSubgraphData:
        out = self.extract(data)
        d = RootedSubgraphData.from_dict(data.to_dict())
        d.sub_edge_index, d.n_id, d.e_id, d.n_sub_batch, d.e_sub_batch = out
        return d


class RootedEgoNets(RootedSubgraph):
    def __init__(self, num_hops: int) -> None:
        super().__init__()
        self.num_hops = num_hops

    def extract(
        self,
        data: Data,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        adj_t = to_torch_csc_tensor(data.edge_index, size=data.size()).t()
        n_mask = torch.eye(num_nodes, device=data.edge_index.device)
        for _ in range(self.num_hops):
            n_mask += adj_t @ n_mask

        return self.map(data, n_mask > 0)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_hops={self.num_hops})'


class RootedRWSubgraph(RootedSubgraph):
    def __init__(self, walk_length: int, repeat: int = 1):
        super().__init__()
        self.walk_length = walk_length
        self.repeat = repeat

    def extract(
        self,
        data: Data,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        from torch_geometric.index import index2ptr
        from torch_geometric.utils import sort_edge_index

        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        start = torch.arange(num_nodes, device=data.edge_index.device)
        start = start.view(-1, 1).repeat(1, self.repeat).view(-1)
        edge_index = sort_edge_index(data.edge_index, num_nodes=num_nodes)
        rowptr = index2ptr(edge_index[0], num_nodes)
        walk = torch.ops.pyg.random_walk(rowptr, edge_index[1], start,
                                         self.walk_length, 1.0, 1.0)

        n_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool,
                             device=walk.device)
        start = start.view(-1, 1).repeat(1, (self.walk_length + 1)).view(-1)
        n_mask[start, walk.view(-1)] = True

        return self.map(data, n_mask)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(walk_length={self.walk_length})'
