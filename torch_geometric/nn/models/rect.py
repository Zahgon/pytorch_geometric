import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from torch_geometric.nn import GCNConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import scatter


class RECT_L(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.conv = GCNConv(in_channels, hidden_channels, normalize=normalize)
        self.lin = Linear(hidden_channels, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight.data)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """"""  # noqa: D419
        x = self.conv(x, edge_index, edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

    @torch.jit.export
    def embed(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        with torch.no_grad():
            return self.conv(x, edge_index, edge_weight)

    @torch.jit.export
    def get_semantic_labels(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
    ) -> Tensor:
        pass

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels})')
