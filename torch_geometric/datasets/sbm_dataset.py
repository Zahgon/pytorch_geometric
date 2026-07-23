import os.path as osp
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import stochastic_blockmodel_graph


class StochasticBlockModelDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        block_sizes: Union[List[int], Tensor],
        edge_probs: Union[List[List[float]], Tensor],
        num_graphs: int = 1,
        num_channels: Optional[int] = None,
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        **kwargs: Any,
    ) -> None:
        if not isinstance(block_sizes, torch.Tensor):
            block_sizes = torch.tensor(block_sizes, dtype=torch.long)
        if not isinstance(edge_probs, torch.Tensor):
            edge_probs = torch.tensor(edge_probs, dtype=torch.float)

        assert num_graphs > 0

        self.block_sizes = block_sizes
        self.edge_probs = edge_probs
        self.num_graphs = num_graphs
        self.num_channels = num_channels
        self.is_undirected = is_undirected

        self.kwargs = {
            'n_informative': num_channels,
            'n_redundant': 0,
            'flip_y': 0.0,
            'shuffle': False,
        }
        self.kwargs.update(kwargs)

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        pass

    @property
    def processed_file_names(self) -> str:
        pass

    def process(self) -> None:
        from sklearn.datasets import make_classification

        edge_index = stochastic_blockmodel_graph(
            self.block_sizes, self.edge_probs, directed=not self.is_undirected)

        num_samples = int(self.block_sizes.sum())
        num_classes = self.block_sizes.size(0)

        data_list = []
        for _ in range(self.num_graphs):
            x = None
            if self.num_channels is not None:
                x, y_not_sorted = make_classification(
                    n_samples=num_samples,
                    n_features=self.num_channels,
                    n_classes=num_classes,
                    weights=self.block_sizes / num_samples,
                    **self.kwargs,
                )
                x = x[np.argsort(y_not_sorted)]
                x = torch.from_numpy(x).to(torch.float)

            y = torch.arange(num_classes).repeat_interleave(self.block_sizes)

            data = Data(x=x, edge_index=edge_index, y=y)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


class RandomPartitionGraphDataset(StochasticBlockModelDataset):
    def __init__(
        self,
        root: str,
        num_classes: int,
        num_nodes_per_class: int,
        node_homophily_ratio: float,
        average_degree: float,
        num_graphs: int = 1,
        num_channels: Optional[int] = None,
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:

        self._num_classes = num_classes
        self.num_nodes_per_class = num_nodes_per_class
        self.node_homophily_ratio = node_homophily_ratio
        self.average_degree = average_degree

        ec_over_v2 = average_degree / num_nodes_per_class
        p_in = node_homophily_ratio * ec_over_v2
        p_out = (ec_over_v2 - p_in) / (num_classes - 1)

        block_sizes = [num_nodes_per_class for _ in range(num_classes)]
        edge_probs = [[p_out for _ in range(num_classes)]
                      for _ in range(num_classes)]
        for r in range(num_classes):
            edge_probs[r][r] = p_in

        super().__init__(root, block_sizes, edge_probs, num_graphs,
                         num_channels, is_undirected, transform, pre_transform,
                         **kwargs)

    @property
    def processed_file_names(self) -> str:
        pass

    def process(self) -> None:
        return super().process()
