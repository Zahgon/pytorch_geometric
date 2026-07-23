from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import cumsum


class QuantileAggregation(Aggregation):
    interpolations = {'linear', 'lower', 'higher', 'nearest', 'midpoint'}

    def __init__(self, q: Union[float, List[float]],
                 interpolation: str = 'linear', fill_value: float = 0.0):
        super().__init__()

        qs = [q] if not isinstance(q, (list, tuple)) else q
        if len(qs) == 0:
            raise ValueError("Provide at least one quantile value for `q`.")
        if not all(0. <= quantile <= 1. for quantile in qs):
            raise ValueError("`q` must be in the range [0, 1].")
        if interpolation not in self.interpolations:
            raise ValueError(f"Invalid interpolation method "
                             f"got ('{interpolation}')")

        self._q = q
        self.register_buffer('q', torch.tensor(qs).view(-1, 1))
        self.interpolation = interpolation
        self.fill_value = fill_value

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        dim = x.dim() + dim if dim < 0 else dim

        self.assert_index_present(index)
        assert index is not None  # Required for TorchScript.

        count = torch.bincount(index, minlength=dim_size or 0)
        ptr = cumsum(count)[:-1]

        if dim_size is not None:
            ptr = ptr.clamp(max=x.size(dim) - 1)

        q_point = self.q * (count - 1) + ptr
        q_point = q_point.t().reshape(-1)

        shape = [1] * x.dim()
        shape[dim] = -1
        index = index.view(shape).expand_as(x)

        x, x_perm = torch.sort(x, dim=dim)
        index = index.take_along_dim(x_perm, dim=dim)
        index, index_perm = torch.sort(index, dim=dim, stable=True)
        x = x.take_along_dim(index_perm, dim=dim)

        if self.interpolation == 'lower':
            quantile = x.index_select(dim, q_point.floor().long())
        elif self.interpolation == 'higher':
            quantile = x.index_select(dim, q_point.ceil().long())
        elif self.interpolation == 'nearest':
            quantile = x.index_select(dim, q_point.round().long())
        else:
            l_quant = x.index_select(dim, q_point.floor().long())
            r_quant = x.index_select(dim, q_point.ceil().long())

            if self.interpolation == 'linear':
                q_frac = q_point.frac().view(shape)
                quantile = l_quant + (r_quant - l_quant) * q_frac
            else:  # 'midpoint'
                quantile = 0.5 * l_quant + 0.5 * r_quant

        repeats = self.q.numel()
        mask = (count == 0).repeat_interleave(
            repeats, output_size=repeats * count.numel()).view(shape)
        out = quantile.masked_fill(mask, self.fill_value)

        if self.q.numel() > 1:
            shape = list(out.shape)
            shape = (shape[:dim] + [shape[dim] // self.q.numel(), -1] +
                     shape[dim + 2:])
            out = out.view(shape)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(q={self._q})')


class MedianAggregation(QuantileAggregation):
    def __init__(self, fill_value: float = 0.0):
        super().__init__(0.5, 'lower', fill_value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
