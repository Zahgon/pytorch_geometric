from typing import Iterator, List, Optional, Tuple, Union

import torch

from torch_geometric.data import Data


def yield_file(in_file: str) -> Iterator[Tuple[str, List[Union[int, float]]]]:
    pass


def read_obj(in_file: str) -> Optional[Data]:
    pass
