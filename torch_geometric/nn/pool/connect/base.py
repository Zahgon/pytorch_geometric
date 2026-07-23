from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.pool.select import SelectOutput


@dataclass(init=False)
class ConnectOutput:
    edge_index: Tensor
    edge_attr: Optional[Tensor] = None
    batch: Optional[Tensor] = None

    def __init__(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ):
        if edge_index.dim() != 2:
            raise ValueError(f"Expected 'edge_index' to be two-dimensional "
                             f"(got {edge_index.dim()} dimensions)")

        if edge_index.size(0) != 2:
            raise ValueError(f"Expected 'edge_index' to have size '2' in the "
                             f"first dimension (got '{edge_index.size(0)}')")

        if edge_attr is not None and edge_attr.size(0) != edge_index.size(1):
            raise ValueError(f"Expected 'edge_index' and 'edge_attr' to "
                             f"hold the same number of edges (got "
                             f"{edge_index.size(1)} and {edge_attr.size(0)} "
                             f"edges)")

        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch


ConnectOutput = torch.jit.script(ConnectOutput)


class Connect(torch.nn.Module):
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""

    def forward(
        self,
        select_output: SelectOutput,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> ConnectOutput:
        r"""Forward pass.

        Args:
            select_output (SelectOutput): The output of :class:`Select`.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
        """
        raise NotImplementedError

    @staticmethod
    def get_pooled_batch(
        select_output: SelectOutput,
        batch: Optional[Tensor],
    ) -> Optional[Tensor]:
        r"""Returns the batch vector of the coarsened graph.

        Args:
            select_output (SelectOutput): The output of :class:`Select`.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
        """
        if batch is None:
            return batch

        out = torch.arange(select_output.num_clusters, device=batch.device)
        return out.scatter_(0, select_output.cluster_index,
                            batch[select_output.node_index])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
