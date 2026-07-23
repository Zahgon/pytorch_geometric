import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('gcn_norm')
class GCNNorm(BaseTransform):
    def __init__(self, add_self_loops: bool = True):
        self.add_self_loops = add_self_loops

    def forward(self, data: Data) -> Data:
        gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm
        assert 'edge_index' in data or 'adj_t' in data

        if 'edge_index' in data:
            data.edge_index, data.edge_weight = gcn_norm(
                data.edge_index, data.edge_weight, data.num_nodes,
                add_self_loops=self.add_self_loops)
        else:
            data.adj_t = gcn_norm(data.adj_t,
                                  add_self_loops=self.add_self_loops)

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'add_self_loops={self.add_self_loops})')
