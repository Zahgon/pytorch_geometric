import torch

from torch_geometric.data import Data
from torch_geometric.io import parse_txt_array
from torch_geometric.utils import coalesce, one_hot

elems = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


def parse_sdf(src: str) -> Data:
    pass


def read_sdf(path: str) -> Data:
    pass
