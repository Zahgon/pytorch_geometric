import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax


class SumAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')


class MeanAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='mean')


class MaxAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='max')


class MinAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='min')


class MulAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        self.assert_index_present(index)
        return self.reduce(x, index, None, dim_size, dim, reduce='mul')


class VarAggregation(Aggregation):
    def __init__(self, semi_grad: bool = False):
        super().__init__()
        self.semi_grad = semi_grad

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        if self.semi_grad:
            with torch.no_grad():
                mean2 = self.reduce(x * x, index, ptr, dim_size, dim, 'mean')
        else:
            mean2 = self.reduce(x * x, index, ptr, dim_size, dim, 'mean')
        return mean2 - mean * mean


class StdAggregation(Aggregation):
    def __init__(self, semi_grad: bool = False):
        super().__init__()
        self.var_aggr = VarAggregation(semi_grad)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        var = self.var_aggr(x, index, ptr, dim_size, dim)
        out = var.clamp(min=1e-5).sqrt()
        out = out.masked_fill(out <= math.sqrt(1e-5), 0.0)
        return out


class SoftmaxAggregation(Aggregation):
    def __init__(self, t: float = 1.0, learn: bool = False,
                 semi_grad: bool = False, channels: int = 1):
        super().__init__()

        if learn and semi_grad:
            raise ValueError(
                f"Cannot enable 'semi_grad' in '{self.__class__.__name__}' in "
                f"case the temperature term 't' is learnable")

        if not learn and channels != 1:
            raise ValueError(f"Cannot set 'channels' greater than '1' in case "
                             f"'{self.__class__.__name__}' is not trainable")

        self._init_t = t
        self.learn = learn
        self.semi_grad = semi_grad
        self.channels = channels

        self.t = Parameter(torch.empty(channels)) if learn else t
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.t, Tensor):
            self.t.data.fill_(self._init_t)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        t = self.t
        if self.channels != 1:
            self.assert_two_dimensional_input(x, dim)
            assert isinstance(t, Tensor)
            t = t.view(-1, self.channels)

        alpha = x
        if not isinstance(t, (int, float)) or t != 1:
            alpha = x * t

        if not self.learn and self.semi_grad:
            with torch.no_grad():
                alpha = softmax(alpha, index, ptr, dim_size, dim)
        else:
            alpha = softmax(alpha, index, ptr, dim_size, dim)
        return self.reduce(x * alpha, index, ptr, dim_size, dim, reduce='sum')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(learn={self.learn})')


class PowerMeanAggregation(Aggregation):
    def __init__(
        self,
        p: float = 1.0,
        learn: bool = False,
        channels: int = 1,
        clamp_min: Optional[float] = 1e-4,
        clamp_max: Optional[float] = 100.,
    ) -> None:
        super().__init__()

        if not learn and channels != 1:
            raise ValueError(f"Cannot set 'channels' greater than '1' in case "
                             f"'{self.__class__.__name__}' is not trainable")

        self._init_p = p
        self.learn = learn
        self.channels = channels

        self.p = Parameter(torch.empty(channels)) if learn else p
        self.reset_parameters()
        self.min_value = clamp_min
        self.max_value = clamp_max

    def reset_parameters(self):
        if isinstance(self.p, Tensor):
            self.p.data.fill_(self._init_p)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        p = self.p
        if self.channels != 1:
            assert isinstance(p, Tensor)
            self.assert_two_dimensional_input(x, dim)
            p = p.view(-1, self.channels)

        if not isinstance(p, (int, float)) or p != 1:
            x = x.clamp(min=self.min_value, max=self.max_value).pow(p)

        out = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')

        if not isinstance(p, (int, float)) or p != 1:
            out = out.clamp(min=self.min_value, max=self.max_value).pow(1. / p)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(learn={self.learn})')
