from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.utils import scatter


class PANPooling(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier

        self.p = Parameter(torch.empty(in_channels))
        self.beta = Parameter(torch.empty(2))
        self.select = SelectTopK(1, ratio, min_score, nonlinearity)
        self.connect = FilterEdges()

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.p.data.fill_(1)
        self.beta.data.fill_(0.5)
        self.select.reset_parameters()

    def forward(
        self,
        x: Tensor,
        M: SparseTensor,
        batch: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, OptTensor, Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node feature matrix.
            M (SparseTensor): The MET matrix :math:`\mathbf{M}`.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        row, col, edge_weight = M.coo()
        assert edge_weight is not None

        score1 = (x * self.p).sum(dim=-1)
        score2 = scatter(edge_weight, col, 0, dim_size=x.size(0), reduce='sum')
        score = self.beta[0] * score1 + self.beta[1] * score2

        select_out = self.select(score, batch)

        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        x = x[perm] * score.view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        edge_index = torch.stack([col, row], dim=0)
        connect_out = self.connect(select_out, edge_index, edge_weight, batch)
        edge_weight = connect_out.edge_attr
        assert edge_weight is not None

        return (x, connect_out.edge_index, edge_weight, connect_out.batch,
                perm, score)

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')
