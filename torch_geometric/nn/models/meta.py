from typing import Optional, Tuple

import torch
from torch import Tensor


class MetaLayer(torch.nn.Module):
    def __init__(
        self,
        edge_model: Optional[torch.nn.Module] = None,
        node_model: Optional[torch.nn.Module] = None,
        global_model: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            u (torch.Tensor, optional): The global graph features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
        """
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u,
                                        batch if batch is None else batch[row])

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')
