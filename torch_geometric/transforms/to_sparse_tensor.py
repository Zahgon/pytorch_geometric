from typing import Optional, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    sort_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


@functional_transform('to_sparse_tensor')
class ToSparseTensor(BaseTransform):
    def __init__(
        self,
        attr: Optional[str] = 'edge_weight',
        remove_edge_index: bool = True,
        fill_cache: bool = True,
        layout: Optional[int] = None,
    ) -> None:
        if layout not in {None, torch.sparse_coo, torch.sparse_csr}:
            raise ValueError(f"Unexpected sparse tensor layout "
                             f"(got '{layout}')")

        self.attr = attr
        self.remove_edge_index = remove_edge_index
        self.fill_cache = fill_cache
        self.layout = layout

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            keys, values = [], []
            for key, value in store.items():
                if key in {'edge_index', 'edge_label', 'edge_label_index'}:
                    continue

                if store.is_edge_attr(key):
                    keys.append(key)
                    values.append(value)

            store.edge_index, values = sort_edge_index(
                store.edge_index,
                values,
                sort_by_row=False,
            )

            for key, value in zip(keys, values):
                store[key] = value

            layout = self.layout
            size = store.size()[::-1]
            edge_weight: Optional[Tensor] = None
            if self.attr is not None and self.attr in store:
                edge_weight = store[self.attr]

            if layout is None and torch_geometric.typing.WITH_TORCH_SPARSE:
                store.adj_t = SparseTensor(
                    row=store.edge_index[1],
                    col=store.edge_index[0],
                    value=edge_weight,
                    sparse_sizes=size,
                    is_sorted=True,
                    trust_data=True,
                )

            elif ((edge_weight is not None and edge_weight.dim() > 1)
                  or layout == torch.sparse_coo):
                assert size[0] is not None and size[1] is not None
                store.adj_t = to_torch_coo_tensor(
                    store.edge_index.flip([0]),
                    edge_attr=edge_weight,
                    size=size,
                )

            elif layout is None or layout == torch.sparse_csr:
                assert size[0] is not None and size[1] is not None
                store.adj_t = to_torch_csr_tensor(
                    store.edge_index.flip([0]),
                    edge_attr=edge_weight,
                    size=size,
                )

            if self.remove_edge_index:
                del store['edge_index']
                if self.attr is not None and self.attr in store:
                    del store[self.attr]

            if self.fill_cache and isinstance(store.adj_t, SparseTensor):
                store.adj_t.storage.rowptr()
                store.adj_t.storage.csr2csc()

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(attr={self.attr}, '
                f'layout={self.layout})')
