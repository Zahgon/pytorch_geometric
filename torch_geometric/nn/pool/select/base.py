from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(init=False)
class SelectOutput:
    node_index: Tensor
    num_nodes: int
    cluster_index: Tensor
    num_clusters: int
    weight: Optional[Tensor] = None

    def __init__(
        self,
        node_index: Tensor,
        num_nodes: int,
        cluster_index: Tensor,
        num_clusters: int,
        weight: Optional[Tensor] = None,
    ):
        if node_index.dim() != 1:
            raise ValueError(f"Expected 'node_index' to be one-dimensional "
                             f"(got {node_index.dim()} dimensions)")

        if cluster_index.dim() != 1:
            raise ValueError(f"Expected 'cluster_index' to be one-dimensional "
                             f"(got {cluster_index.dim()} dimensions)")

        if node_index.numel() != cluster_index.numel():
            raise ValueError(f"Expected 'node_index' and 'cluster_index' to "
                             f"hold the same number of values (got "
                             f"{node_index.numel()} and "
                             f"{cluster_index.numel()} values)")

        if weight is not None and weight.dim() != 1:
            raise ValueError(f"Expected 'weight' vector to be one-dimensional "
                             f"(got {weight.dim()} dimensions)")

        if weight is not None and weight.numel() != node_index.numel():
            raise ValueError(f"Expected 'weight' to hold {node_index.numel()} "
                             f"values (got {weight.numel()} values)")

        self.node_index = node_index
        self.num_nodes = num_nodes
        self.cluster_index = cluster_index
        self.num_clusters = num_clusters
        self.weight = weight


SelectOutput = torch.jit.script(SelectOutput)


class Select(torch.nn.Module):
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""

    def forward(self, *args, **kwargs) -> SelectOutput:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
