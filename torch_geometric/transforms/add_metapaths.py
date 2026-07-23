import warnings
from typing import List, Optional, Tuple, Union, cast

import torch
from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType
from torch_geometric.utils import coalesce, degree


@functional_transform('add_metapaths')
class AddMetaPaths(BaseTransform):
    def __init__(
        self,
        metapaths: List[List[EdgeType]],
        drop_orig_edge_types: bool = False,
        keep_same_node_type: bool = False,
        drop_unconnected_node_types: bool = False,
        max_sample: Optional[int] = None,
        weighted: bool = False,
        **kwargs: bool,
    ) -> None:
        if 'drop_orig_edges' in kwargs:
            warnings.warn(
                "'drop_orig_edges' is deprecated. Use "
                "'drop_orig_edge_types' instead", stacklevel=2)
            drop_orig_edge_types = kwargs['drop_orig_edges']

        if 'drop_unconnected_nodes' in kwargs:
            warnings.warn(
                "'drop_unconnected_nodes' is deprecated. Use "
                "'drop_unconnected_node_types' instead", stacklevel=2)
            drop_unconnected_node_types = kwargs['drop_unconnected_nodes']

        for path in metapaths:
            assert len(path) >= 2, f"Invalid metapath '{path}'"
            assert all([
                j[-1] == path[i + 1][0] for i, j in enumerate(path[:-1])
            ]), f"Invalid sequence of node types in '{path}'"

        self.metapaths = metapaths
        self.drop_orig_edge_types = drop_orig_edge_types
        self.keep_same_node_type = keep_same_node_type
        self.drop_unconnected_node_types = drop_unconnected_node_types
        self.max_sample = max_sample
        self.weighted = weighted

    def forward(self, data: HeteroData) -> HeteroData:
        edge_types = data.edge_types  # Save original edge types.
        data.metapath_dict = {}

        for j, metapath in enumerate(self.metapaths):
            for edge_type in metapath:
                assert data._to_canonical(edge_type) in edge_types

            edge_type = metapath[0]
            edge_index, edge_weight = self._edge_index(data, edge_type)

            if self.max_sample is not None:
                edge_index, edge_weight = self._sample(edge_index, edge_weight)

            for edge_type in metapath[1:]:
                edge_index2, edge_weight2 = self._edge_index(data, edge_type)

                edge_index, edge_weight = edge_index.matmul(
                    edge_index2, edge_weight, edge_weight2)

                if not self.weighted:
                    edge_weight = None

                if self.max_sample is not None:
                    edge_index, edge_weight = self._sample(
                        edge_index, edge_weight)

            new_edge_type = (metapath[0][0], f'metapath_{j}', metapath[-1][-1])
            data[new_edge_type].edge_index = edge_index.as_tensor()
            if self.weighted:
                data[new_edge_type].edge_weight = edge_weight
            data.metapath_dict[new_edge_type] = metapath

        postprocess(data, edge_types, self.drop_orig_edge_types,
                    self.keep_same_node_type, self.drop_unconnected_node_types)

        return data

    def _edge_index(
        self,
        data: HeteroData,
        edge_type: EdgeType,
    ) -> Tuple[EdgeIndex, Optional[Tensor]]:

        edge_index = EdgeIndex(
            data[edge_type].edge_index,
            sparse_size=data[edge_type].size(),
        )
        edge_index, perm = edge_index.sort_by('row')

        if not self.weighted:
            return edge_index, None

        edge_weight = data[edge_type].get('edge_weight')
        if edge_weight is not None:
            assert edge_weight.dim() == 1
            edge_weight = edge_weight[perm]

        return edge_index, edge_weight

    def _sample(
        self,
        edge_index: EdgeIndex,
        edge_weight: Optional[Tensor],
    ) -> Tuple[EdgeIndex, Optional[Tensor]]:
        assert self.max_sample is not None

        deg = degree(edge_index[0], num_nodes=edge_index.get_sparse_size(0))
        prob = (self.max_sample * (1. / deg))[edge_index[0]]
        mask = torch.rand_like(prob) < prob

        edge_index = cast(EdgeIndex, edge_index[:, mask])
        assert isinstance(edge_index, EdgeIndex)
        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return edge_index, edge_weight


@functional_transform('add_random_metapaths')
class AddRandomMetaPaths(BaseTransform):
    def __init__(
        self,
        metapaths: List[List[EdgeType]],
        drop_orig_edge_types: bool = False,
        keep_same_node_type: bool = False,
        drop_unconnected_node_types: bool = False,
        walks_per_node: Union[int, List[int]] = 1,
        sample_ratio: float = 1.0,
    ):

        for path in metapaths:
            assert len(path) >= 2, f"Invalid metapath '{path}'"
            assert all([
                j[-1] == path[i + 1][0] for i, j in enumerate(path[:-1])
            ]), f"Invalid sequence of node types in '{path}'"

        self.metapaths = metapaths
        self.drop_orig_edge_types = drop_orig_edge_types
        self.keep_same_node_type = keep_same_node_type
        self.drop_unconnected_node_types = drop_unconnected_node_types
        self.sample_ratio = sample_ratio
        if isinstance(walks_per_node, int):
            walks_per_node = [walks_per_node] * len(metapaths)
        assert len(walks_per_node) == len(metapaths)
        self.walks_per_node = walks_per_node

    def forward(self, data: HeteroData) -> HeteroData:
        edge_types = data.edge_types  # save original edge types
        data.metapath_dict = {}

        for j, metapath in enumerate(self.metapaths):
            for edge_type in metapath:
                assert data._to_canonical(
                    edge_type) in edge_types, f"'{edge_type}' not present"

            src_node = metapath[0][0]
            num_nodes = data[src_node].num_nodes
            num_starts = round(num_nodes * self.sample_ratio)
            row = start = torch.randperm(num_nodes)[:num_starts].repeat(
                self.walks_per_node[j])

            for edge_type in metapath:
                edge_index = EdgeIndex(
                    data[edge_type].edge_index,
                    sparse_size=data[edge_type].size(),
                )
                col, mask = self.sample(edge_index, start)
                row, col = row[mask], col[mask]
                start = col

            new_edge_type = (metapath[0][0], f'metapath_{j}', metapath[-1][-1])
            data[new_edge_type].edge_index = coalesce(torch.vstack([row, col]))
            data.metapath_dict[new_edge_type] = metapath

        postprocess(data, edge_types, self.drop_orig_edge_types,
                    self.keep_same_node_type, self.drop_unconnected_node_types)

        return data

    @staticmethod
    def sample(edge_index: EdgeIndex, subset: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample neighbors from :obj:`edge_index` for each node in
        :obj:`subset`.
        """
        edge_index, _ = edge_index.sort_by('row')
        rowptr = edge_index.get_indptr()
        rowcount = rowptr.diff()[subset]

        mask = rowcount > 0
        offset = torch.zeros_like(subset)
        offset[mask] = rowptr[subset[mask]]

        rand = torch.rand((rowcount.size(0), 1), device=subset.device)
        rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
        rand = rand.to(torch.long)
        rand.add_(offset.view(-1, 1))
        col = edge_index[1][rand].squeeze()
        return col, mask

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'sample_ratio={self.sample_ratio}, '
                f'walks_per_node={self.walks_per_node})')


def postprocess(
    data: HeteroData,
    edge_types: List[EdgeType],
    drop_orig_edge_types: bool,
    keep_same_node_type: bool,
    drop_unconnected_node_types: bool,
) -> None:

    if drop_orig_edge_types:
        for i in edge_types:
            if keep_same_node_type and i[0] == i[-1]:
                continue
            else:
                del data[i]

    if drop_unconnected_node_types:
        new_edge_types = data.edge_types
        node_types = data.node_types
        connected_nodes = set()
        for i in new_edge_types:
            connected_nodes.add(i[0])
            connected_nodes.add(i[-1])
        for node in node_types:
            if node not in connected_nodes:
                del data[node]
