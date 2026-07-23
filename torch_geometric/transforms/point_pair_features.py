import torch

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('point_pair_features')
class PointPairFeatures(BaseTransform):
    def __init__(self, cat: bool = True):
        self.cat = cat

    def forward(self, data: Data) -> Data:
        ppf_func = torch_geometric.nn.conv.ppf_conv.point_pair_features

        assert data.edge_index is not None
        assert data.pos is not None and data.norm is not None
        assert data.pos.size(-1) == 3
        assert data.pos.size() == data.norm.size()

        row, col = data.edge_index
        pos, norm, pseudo = data.pos, data.norm, data.edge_attr

        ppf = ppf_func(pos[row], pos[col], norm[row], norm[col])

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, ppf.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = ppf

        return data
