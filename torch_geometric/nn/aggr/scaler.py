from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.utils import degree


class DegreeScalerAggregation(Aggregation):
    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation],
        scaler: Union[str, List[str]],
        deg: Tensor,
        train_norm: bool = False,
        aggr_kwargs: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()

        if isinstance(aggr, (str, Aggregation)):
            self.aggr = aggr_resolver(aggr, **(aggr_kwargs or {}))
        elif isinstance(aggr, (tuple, list)):
            self.aggr = MultiAggregation(aggr, aggr_kwargs)
        else:
            raise ValueError(f"Only strings, list, tuples and instances of"
                             f"`torch_geometric.nn.aggr.Aggregation` are "
                             f"valid aggregation schemes (got '{type(aggr)}')")

        self.scaler = [scaler] if isinstance(aggr, str) else scaler

        deg = deg.to(torch.float)
        N = int(deg.sum())
        bin_degree = torch.arange(deg.numel(), device=deg.device)

        self.init_avg_deg_lin = float((bin_degree * deg).sum()) / N
        self.init_avg_deg_log = float(((bin_degree + 1).log() * deg).sum()) / N

        if train_norm:
            self.avg_deg_lin = torch.nn.Parameter(torch.empty(1))
            self.avg_deg_log = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('avg_deg_lin', torch.empty(1))
            self.register_buffer('avg_deg_log', torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.avg_deg_lin.data.fill_(self.init_avg_deg_lin)
        self.avg_deg_log.data.fill_(self.init_avg_deg_log)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        self.assert_index_present(index)

        out = self.aggr(x, index, ptr, dim_size, dim)

        assert index is not None
        deg = degree(index, num_nodes=dim_size, dtype=out.dtype)
        size = [1] * len(out.size())
        size[dim] = -1
        deg = deg.view(size)

        outs = []
        for scaler in self.scaler:
            if scaler == 'identity':
                out_scaler = out
            elif scaler == 'amplification':
                out_scaler = out * (torch.log(deg + 1) / self.avg_deg_log)
            elif scaler == 'attenuation':
                out_scaler = out * (self.avg_deg_log /
                                    torch.log(deg.clamp(min=1) + 1))
            elif scaler == 'linear':
                out_scaler = out * (deg / self.avg_deg_lin)
            elif scaler == 'inverse_linear':
                out_scaler = out * (self.avg_deg_lin / deg.clamp(min=1))
            else:
                raise ValueError(f"Unknown scaler '{scaler}'")
            outs.append(out_scaler)

        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
