from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm
from typing_extensions import Self

from torch_geometric.data import Dataset, HeteroData
from torch_geometric.typing import EdgeType, NodeType


@dataclass
class Stats:
    mean: float
    std: float
    min: float
    quantile25: float
    median: float
    quantile75: float
    max: float

    @classmethod
    def from_data(
        cls,
        data: Union[List[int], List[float], torch.Tensor],
    ) -> Self:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        data = data.to(torch.float)

        return cls(
            mean=data.mean().item(),
            std=data.std().item(),
            min=data.min().item(),
            quantile25=data.quantile(0.25).item(),
            median=data.median().item(),
            quantile75=data.quantile(0.75).item(),
            max=data.max().item(),
        )


@dataclass(repr=False)
class Summary:
    name: str
    num_graphs: int
    num_nodes: Stats
    num_edges: Stats
    num_nodes_per_type: Optional[Dict[NodeType, Stats]] = None
    num_edges_per_type: Optional[Dict[EdgeType, Stats]] = None

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        progress_bar: Optional[bool] = None,
        per_type: bool = True,
    ) -> Self:
        pass

    def format(self, fmt: str = "psql") -> str:
        pass

    def __repr__(self) -> str:
        return self.format()
