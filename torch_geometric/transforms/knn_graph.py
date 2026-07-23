import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected


@functional_transform('knn_graph')
class KNNGraph(BaseTransform):
    def __init__(
        self,
        k: int = 6,
        loop: bool = False,
        force_undirected: bool = False,
        flow: str = 'source_to_target',
        cosine: bool = False,
        num_workers: int = 1,
    ) -> None:
        self.k = k
        self.loop = loop
        self.force_undirected = force_undirected
        self.flow = flow
        self.cosine = cosine
        self.num_workers = num_workers

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        edge_index = torch_geometric.nn.knn_graph(
            data.pos,
            self.k,
            data.batch,
            loop=self.loop,
            flow=self.flow,
            cosine=self.cosine,
            num_workers=self.num_workers,
        )

        if self.force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.edge_index = edge_index
        data.edge_attr = None

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'
