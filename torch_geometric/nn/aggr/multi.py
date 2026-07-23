import copy
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear, MultiheadAttention

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.nn.dense import HeteroDictLinear
from torch_geometric.nn.resolver import aggregation_resolver


class MultiAggregation(Aggregation):
    fused_out_index: List[int]
    is_fused_aggr: List[bool]

    def __init__(
        self,
        aggrs: List[Union[Aggregation, str]],
        aggrs_kwargs: Optional[List[Dict[str, Any]]] = None,
        mode: Optional[str] = 'cat',
        mode_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super().__init__()

        if not isinstance(aggrs, (list, tuple)):
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should "
                             f"be a list or tuple (got '{type(aggrs)}').")

        if len(aggrs) == 0:
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should "
                             f"not be empty.")

        if aggrs_kwargs is None:
            aggrs_kwargs = [{}] * len(aggrs)
        elif len(aggrs) != len(aggrs_kwargs):
            raise ValueError(f"'aggrs_kwargs' with invalid length passed to "
                             f"'{self.__class__.__name__}' "
                             f"(got '{len(aggrs_kwargs)}', "
                             f"expected '{len(aggrs)}'). Ensure that both "
                             f"'aggrs' and 'aggrs_kwargs' are consistent.")

        self.aggrs = torch.nn.ModuleList([
            aggregation_resolver(aggr, **aggr_kwargs)
            for aggr, aggr_kwargs in zip(aggrs, aggrs_kwargs)
        ])

        fused_aggrs: List[Aggregation] = []
        self.fused_out_index: List[int] = []
        self.is_fused_aggr: List[bool] = []
        for i, aggr in enumerate(self.aggrs):
            if aggr.__class__ in FusedAggregation.FUSABLE_AGGRS:
                fused_aggrs.append(aggr)
                self.fused_out_index.append(i)
                self.is_fused_aggr.append(True)
            else:
                self.is_fused_aggr.append(False)

        if len(fused_aggrs) > 0:
            self.fused_aggr = FusedAggregation(fused_aggrs)
        else:
            self.fused_aggr = None

        self.mode = mode
        mode_kwargs = copy.copy(mode_kwargs) or {}

        self.in_channels = mode_kwargs.pop('in_channels', None)
        self.out_channels = mode_kwargs.pop('out_channels', None)

        if mode == 'proj' or mode == 'attn':
            if len(aggrs) == 1:
                raise ValueError("Multiple aggregations are required for "
                                 "'proj' or 'attn' combine mode.")

            if (self.in_channels and self.out_channels) is None:
                raise ValueError(
                    f"Combine mode '{mode}' must have `in_channels` "
                    f"and `out_channels` specified.")

            if isinstance(self.in_channels, int):
                self.in_channels = [self.in_channels] * len(aggrs)

            if mode == 'proj':
                self.lin = Linear(
                    sum(self.in_channels),
                    self.out_channels,
                    **mode_kwargs,
                )

            elif mode == 'attn':
                channels = {str(k): v for k, v, in enumerate(self.in_channels)}
                self.lin_heads = HeteroDictLinear(channels, self.out_channels)
                num_heads = mode_kwargs.pop('num_heads', 1)
                self.multihead_attn = MultiheadAttention(
                    self.out_channels,
                    num_heads,
                    **mode_kwargs,
                )

        dense_combine_modes = [
            'sum', 'mean', 'max', 'min', 'logsumexp', 'std', 'var'
        ]
        if mode in dense_combine_modes:
            self.dense_combine = getattr(torch, mode)

    def reset_parameters(self):
        for aggr in self.aggrs:
            aggr.reset_parameters()
        if self.mode == 'proj':
            self.lin.reset_parameters()
        if self.mode == 'attn':
            self.lin_heads.reset_parameters()
            self.multihead_attn._reset_parameters()

    def get_out_channels(self, in_channels: int) -> int:
        if self.out_channels is not None:
            return self.out_channels
        if self.mode == 'cat':
            return in_channels * len(self.aggrs)
        return in_channels

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        if index is None or x.dim() != 2 or self.fused_aggr is None:
            outs = [aggr(x, index, ptr, dim_size, dim) for aggr in self.aggrs]
            return self.combine(outs)

        outs: List[Tensor] = [x] * len(self.aggrs)  # Fill with dummy tensors.

        fused_outs = self.fused_aggr(x, index, ptr, dim_size, dim)
        for i, out in zip(self.fused_out_index, fused_outs):
            outs[i] = out

        for i, aggr in enumerate(self.aggrs):
            if not self.is_fused_aggr[i]:
                outs[i] = aggr(x, index, ptr, dim_size, dim)

        return self.combine(outs)

    def combine(self, inputs: List[Tensor]) -> Tensor:
        if len(inputs) == 1:
            return inputs[0]

        if self.mode == 'cat':
            return torch.cat(inputs, dim=-1)

        if hasattr(self, 'lin'):
            return self.lin(torch.cat(inputs, dim=-1))

        if hasattr(self, 'multihead_attn'):
            x_dict = {str(k): v for k, v, in enumerate(inputs)}
            x_dict = self.lin_heads(x_dict)
            xs = [x_dict[str(key)] for key in range(len(inputs))]
            x = torch.stack(xs, dim=0)
            attn_out, _ = self.multihead_attn(x, x, x)
            return torch.mean(attn_out, dim=0)

        if hasattr(self, 'dense_combine'):
            out = self.dense_combine(torch.stack(inputs, dim=0), dim=0)
            return out if isinstance(out, Tensor) else out[0]

        raise ValueError(f"Combine mode '{self.mode}' is not supported.")

    def __repr__(self) -> str:
        aggrs = ',\n'.join([f'  {aggr}' for aggr in self.aggrs]) + ',\n'
        return f'{self.__class__.__name__}([\n{aggrs}], mode={self.mode})'
