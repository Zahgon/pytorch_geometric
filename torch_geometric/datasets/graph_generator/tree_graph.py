from typing import List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.utils import to_undirected


def tree(
    depth: int,
    branch: int = 2,
    undirected: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    pass


class TreeGraph(GraphGenerator):
    def __init__(
        self,
        depth: int,
        branch: int = 2,
        undirected: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.branch = branch
        self.undirected = undirected

    def __call__(self) -> Data:
        edge_index, depth = tree(self.depth, self.branch, self.undirected)
        num_nodes = depth.numel()
        return Data(edge_index=edge_index, depth=depth, num_nodes=num_nodes)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(depth={self.depth}, '
                f'branch={self.branch}, undirected={self.undirected})')
