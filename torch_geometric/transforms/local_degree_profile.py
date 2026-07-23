import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


@functional_transform('local_degree_profile')
class LocalDegreeProfile(BaseTransform):
    def __init__(self) -> None:
        from torch_geometric.nn.aggr.fused import FusedAggregation
        self.aggr = FusedAggregation(['min', 'max', 'mean', 'std'])

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        num_nodes = data.num_nodes

        deg = degree(row, num_nodes, dtype=torch.float).view(-1, 1)
        xs = [deg] + self.aggr(deg[col], row, dim_size=num_nodes)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x] + xs, dim=-1)
        else:
            data.x = torch.cat(xs, dim=-1)

        return data
