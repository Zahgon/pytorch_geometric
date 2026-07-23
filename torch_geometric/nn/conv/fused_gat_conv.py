from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.index import index2ptr
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import sort_edge_index


class FusedGATConv(GATConv):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.add_self_loops:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops. Please add them manually "
                             f"in a pre-processing step and set "
                             f"`add_self_loops=False`.")

        if self.edge_dim is not None:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"edge features. Set `edge_dim=None` in order "
                             f"to proceed.")

        from dgNN.operators import GATConvFuse
        self.op = GATConvFuse

    @staticmethod
    def to_graph_format(
        edge_index: Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor]:
        pass

    def forward(
        self,
        x: Tensor,
        csr: Tuple[Tensor, Tensor],
        csc: Tuple[Tensor, Tensor],
        perm: Tensor,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The node features.
            csr ((torch.Tensor, torch.Tensor)): A tuple containing the CSR
                representation of a graph, given as a tuple of
                :obj:`(rowptr, col)`.
            csc ((torch.Tensor, torch.Tensor)): A tuple containing the CSC
                representation of a graph, given as a tuple of
                :obj:`(row, colptr)`.
            perm (torch.Tensor): Permutation tensor to map the CSR
                representation to the CSC representation.

        .. note::

            Use the
            :meth:`~torch_geometric.nn.conv.FusedGATConv.to_graph_format`
            method to obtain the :obj:`(csr, csc, perm)` graph format from an
            existing :obj:`edge_index` representation.
        """
        H, C = self.heads, self.out_channels

        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        x = self.lin_src(x).view(-1, H, C)

        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)

        dropout = self.dropout if self.training else 0.0

        (rowptr, col), (row, colptr) = csr, csc
        out = self.op(alpha_dst, alpha_src, rowptr, col, colptr, row, perm,
                      self.negative_slope, x, dropout)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out
