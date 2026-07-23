from typing import Any

from torch.nn import Module

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, VirtualNode


@functional_transform('add_gpse')
class AddGPSE(BaseTransform):
    def __init__(
        self,
        model: Module,
        use_vn: bool = True,
        rand_type: str = 'NormalSE',
    ):
        self.model = model
        self.use_vn = use_vn
        self.vn = VirtualNode()
        self.rand_type = rand_type

    def forward(self, data: Data) -> Any:
        pass

    def __call__(self, data: Data) -> Data:
        from torch_geometric.nn.models.gpse import gpse_process

        data_vn = self.vn(data.clone()) if self.use_vn else data.clone()
        batch_out = gpse_process(self.model, data_vn, 'NormalSE', self.use_vn)
        batch_out = batch_out.to('cpu', non_blocking=True)
        data.pestat_GPSE = batch_out[:-1] if self.use_vn else batch_out

        return data
