from typing import Union

from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops


@functional_transform('add_self_loops')
class AddSelfLoops(BaseTransform):
    def __init__(
        self,
        attr: str = 'edge_weight',
        fill_value: Union[float, Tensor, str] = 1.0,
    ) -> None:
        self.attr = attr
        self.fill_value = fill_value

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            store.edge_index, store[self.attr] = add_self_loops(
                store.edge_index,
                edge_attr=store.get(self.attr, None),
                fill_value=self.fill_value,
                num_nodes=store.size(0),
            )

        return data
