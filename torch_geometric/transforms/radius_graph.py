import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('radius_graph')
class RadiusGraph(BaseTransform):
    def __init__(
        self,
        r: float,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = 'source_to_target',
        num_workers: int = 1,
    ) -> None:
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.num_workers = num_workers

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        data.edge_index = torch_geometric.nn.radius_graph(
            data.pos,
            self.r,
            data.batch,
            self.loop,
            max_num_neighbors=self.max_num_neighbors,
            flow=self.flow,
            num_workers=self.num_workers,
        )
        data.edge_attr = None

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'
