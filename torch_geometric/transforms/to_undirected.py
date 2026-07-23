from typing import Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected


@functional_transform('to_undirected')
class ToUndirected(BaseTransform):
    def __init__(self, reduce: str = "add", merge: bool = True):
        self.reduce = reduce
        self.merge = merge

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            nnz = store.edge_index.size(1)

            if isinstance(data, HeteroData) and (store.is_bipartite()
                                                 or not self.merge):
                src, rel, dst = store._key

                row, col = store.edge_index
                rev_edge_index = torch.stack([col, row], dim=0)

                inv_store = data[dst, f'rev_{rel}', src]
                inv_store.edge_index = rev_edge_index
                for key, value in store.items():
                    if key == 'edge_index':
                        continue
                    if isinstance(value, Tensor) and value.size(0) == nnz:
                        inv_store[key] = value

            else:
                keys, values = [], []
                for key, value in store.items():
                    if key == 'edge_index':
                        continue

                    if store.is_edge_attr(key):
                        keys.append(key)
                        values.append(value)

                store.edge_index, values = to_undirected(
                    store.edge_index, values, reduce=self.reduce)

                for key, value in zip(keys, values):
                    store[key] = value

        return data
