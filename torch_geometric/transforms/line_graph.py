import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce, cumsum, remove_self_loops, scatter


@functional_transform('line_graph')
class LineGraph(BaseTransform):
    def __init__(self, force_directed: bool = False) -> None:
        self.force_directed = force_directed

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=N)
        row, col = edge_index

        if self.force_directed or data.is_directed():
            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

            count = scatter(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes, reduce='sum')
            ptr = cumsum(count)

            cols = [i[ptr[col[j]]:ptr[col[j] + 1]] for j in range(col.size(0))]
            rows = [row.new_full((c.numel(), ), j) for j, c in enumerate(cols)]

            row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)

            data.edge_index = torch.stack([row, col], dim=0)
            data.x = data.edge_attr
            data.num_nodes = edge_index.size(1)

        else:
            mask = row < col
            row, col = row[mask], col[mask]
            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

            (row, col), i = coalesce(
                torch.stack([
                    torch.cat([row, col], dim=0),
                    torch.cat([col, row], dim=0)
                ], dim=0),
                torch.cat([i, i], dim=0),
                N,
            )

            count = scatter(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes, reduce='sum')
            joints = list(torch.split(i, count.tolist()))

            def generate_grid(x: Tensor) -> Tensor:
                row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
                col = x.repeat(x.numel())
                return torch.stack([row, col], dim=0)

            joints = [generate_grid(joint) for joint in joints]
            joint = torch.cat(joints, dim=1)
            joint, _ = remove_self_loops(joint)
            N = row.size(0) // 2
            joint = coalesce(joint, num_nodes=N)

            if edge_attr is not None:
                data.x = scatter(edge_attr, i, dim=0, dim_size=N, reduce='sum')
            data.edge_index = joint
            data.num_nodes = edge_index.size(1) // 2

        data.edge_attr = None

        return data
