from typing import List, Optional, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce


@functional_transform('remove_duplicated_edges')
class RemoveDuplicatedEdges(BaseTransform):
    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        reduce: str = "add",
    ) -> None:
        key = key or ['edge_attr', 'edge_weight']

        if isinstance(key, str):
            key = [key]

        self.keys = key
        self.reduce = reduce

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.edge_stores:
            keys = [key for key in self.keys if key in store]

            size = [s for s in store.size() if s is not None]
            num_nodes = max(size) if len(size) > 0 else None

            store.edge_index, edge_attrs = coalesce(
                edge_index=store.edge_index,
                edge_attr=[store[key] for key in keys],
                num_nodes=num_nodes,
                reduce=self.reduce,
            )

            for key, edge_attr in zip(keys, edge_attrs):
                store[key] = edge_attr

        return data
