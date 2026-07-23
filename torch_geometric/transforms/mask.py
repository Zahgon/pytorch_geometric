from typing import List, Optional, Sequence, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data.storage import BaseStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import index_to_mask, mask_to_index

AnyData = Union[Data, HeteroData]


def get_attrs_with_suffix(
    attrs: Optional[List[str]],
    store: BaseStorage,
    suffix: str,
) -> List[str]:
    if attrs is not None:
        return attrs
    return [key for key in store.keys() if key.endswith(suffix)]


def get_mask_size(
    attr: str,
    store: BaseStorage,
    size: Optional[int],
) -> Optional[int]:
    if size is not None:
        return size
    return store.num_edges if store.is_edge_attr(attr) else store.num_nodes


@functional_transform('index_to_mask')
class IndexToMask(BaseTransform):
    def __init__(
        self,
        attrs: Optional[Union[str, List[str]]] = None,
        sizes: Optional[Union[int, List[int]]] = None,
        replace: bool = False,
    ) -> None:
        self.attrs = [attrs] if isinstance(attrs, str) else attrs
        self.sizes = sizes
        self.replace = replace

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            attrs = get_attrs_with_suffix(self.attrs, store, '_index')

            sizes: Sequence[Optional[int]]
            if isinstance(self.sizes, int):
                sizes = [self.sizes] * len(attrs)
            elif isinstance(self.sizes, (list, tuple)):
                if len(attrs) != len(self.sizes):
                    raise ValueError(
                        f"The number of attributes (got {len(attrs)}) must "
                        f"match the number of sizes provided "
                        f"(got {len(self.sizes)})")
                sizes = self.sizes
            else:
                sizes = [None] * len(attrs)

            for attr, size in zip(attrs, sizes):
                if 'edge_index' in attr:
                    continue
                if attr not in store:
                    continue
                size = get_mask_size(attr, store, size)
                mask = index_to_mask(store[attr], size=size)
                store[f'{attr[:-6]}_mask'] = mask
                if self.replace:
                    del store[attr]

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(attrs={self.attrs}, '
                f'sizes={self.sizes}, replace={self.replace})')


@functional_transform('mask_to_index')
class MaskToIndex(BaseTransform):
    def __init__(
        self,
        attrs: Optional[Union[str, List[str]]] = None,
        replace: bool = False,
    ):
        self.attrs = [attrs] if isinstance(attrs, str) else attrs
        self.replace = replace

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            attrs = get_attrs_with_suffix(self.attrs, store, '_mask')

            for attr in attrs:
                if attr not in store:
                    continue
                index = mask_to_index(store[attr])
                store[f'{attr[:-5]}_index'] = index
                if self.replace:
                    del store[attr]

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(attrs={self.attrs}, '
                f'replace={self.replace})')
