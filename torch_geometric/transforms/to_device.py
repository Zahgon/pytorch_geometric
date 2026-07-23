from typing import List, Optional, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('to_device')
class ToDevice(BaseTransform):
    def __init__(
        self,
        device: Union[int, str],
        attrs: Optional[List[str]] = None,
        non_blocking: bool = False,
    ) -> None:
        self.device = device
        self.attrs = attrs or []
        self.non_blocking = non_blocking

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        return data.to(self.device, *self.attrs,
                       non_blocking=self.non_blocking)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.device})'
