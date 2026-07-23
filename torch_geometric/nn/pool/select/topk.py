from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.inits import uniform
from torch_geometric.nn.pool.select import Select, SelectOutput
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import cumsum, scatter, softmax


def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")


class SelectTopK(Select):
    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        act: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        if ratio is None and min_score is None:
            raise ValueError(f"At least one of the 'ratio' and 'min_score' "
                             f"parameters must be specified in "
                             f"'{self.__class__.__name__}'")

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.act = activation_resolver(act)

        self.weight = torch.nn.Parameter(torch.empty(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
    ) -> SelectOutput:
        """"""  # noqa: D419
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = x.view(-1, 1) if x.dim() == 1 else x
        score = (x * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.act(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        node_index = topk(score, self.ratio, batch, self.min_score)

        return SelectOutput(
            node_index=node_index,
            num_nodes=x.size(0),
            cluster_index=torch.arange(node_index.size(0), device=x.device),
            num_clusters=node_index.size(0),
            weight=score[node_index],
        )

    def __repr__(self) -> str:
        if self.min_score is None:
            arg = f'ratio={self.ratio}'
        else:
            arg = f'min_score={self.min_score}'
        return f'{self.__class__.__name__}({self.in_channels}, {arg})'
