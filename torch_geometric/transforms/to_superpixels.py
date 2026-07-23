from typing import Any

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter


@functional_transform('to_slic')
class ToSLIC(BaseTransform):
    def __init__(
        self,
        add_seg: bool = False,
        add_img: bool = False,
        **kwargs: Any,
    ) -> None:
        self.add_seg = add_seg
        self.add_img = add_img
        self.kwargs = kwargs

    def forward(self, img: Tensor) -> Data:
        from skimage.segmentation import slic

        img = img.permute(1, 2, 0)
        h, w, c = img.size()

        seg = slic(img.to(torch.double).numpy(), start_label=0, **self.kwargs)
        seg = torch.from_numpy(seg)

        x = scatter(img.view(h * w, c), seg.view(h * w), dim=0, reduce='mean')

        pos_y = torch.arange(h, dtype=torch.float)
        pos_y = pos_y.view(-1, 1).repeat(1, w).view(h * w)
        pos_x = torch.arange(w, dtype=torch.float)
        pos_x = pos_x.view(1, -1).repeat(h, 1).view(h * w)

        pos = torch.stack([pos_x, pos_y], dim=-1)
        pos = scatter(pos, seg.view(h * w), dim=0, reduce='mean')

        data = Data(x=x, pos=pos)

        if self.add_seg:
            data.seg = seg.view(1, h, w)

        if self.add_img:
            data.img = img.permute(2, 0, 1).view(1, c, h, w)

        return data
