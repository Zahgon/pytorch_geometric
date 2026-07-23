import torch

from torch_geometric.data import Data
from torch_geometric.datasets.motif_generator import CustomMotif


class HouseMotif(CustomMotif):
    def __init__(self) -> None:
        structure = Data(
            num_nodes=5,
            edge_index=torch.tensor([
                [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1],
            ]),
            y=torch.tensor([0, 0, 1, 1, 2]),
        )
        super().__init__(structure)
