import math
from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.aggr.utils import MultiheadAttentionBlock
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.utils import scatter


class PatchTransformerAggregation(Aggregation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        hidden_channels: int,
        num_transformer_blocks: int = 1,
        heads: int = 1,
        dropout: float = 0.0,
        aggr: Union[str, List[str]] = 'mean',
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.aggrs = [aggr] if isinstance(aggr, str) else aggr

        assert len(self.aggrs) > 0
        for aggr in self.aggrs:
            assert aggr in ['sum', 'mean', 'min', 'max', 'var', 'std']

        self.lin = torch.nn.Linear(in_channels, hidden_channels, device=device)
        self.pad_projector = torch.nn.Linear(
            patch_size * hidden_channels,
            hidden_channels,
            device=device,
        )
        self.pe = PositionalEncoding(hidden_channels, device=device)

        self.blocks = torch.nn.ModuleList([
            MultiheadAttentionBlock(
                channels=hidden_channels,
                heads=heads,
                layer_norm=True,
                dropout=dropout,
                device=device,
            ) for _ in range(num_transformer_blocks)
        ])

        self.fc = torch.nn.Linear(
            hidden_channels * len(self.aggrs),
            out_channels,
            device=device,
        )

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self.pad_projector.reset_parameters()
        self.pe.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        self.fc.reset_parameters()

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
        self,
        x: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:

        if max_num_elements is None:
            if ptr is not None:
                count = ptr.diff()
            else:
                count = scatter(torch.ones_like(index), index, dim=0,
                                dim_size=dim_size, reduce='sum')
            max_num_elements = int(count.max()) + 1

        max_num_elements = (math.floor(max_num_elements / self.patch_size) *
                            self.patch_size)

        x = self.lin(x)

        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=max_num_elements)

        x = x.view(x.size(0), max_num_elements // self.patch_size,
                   self.patch_size * x.size(-1))

        x = self.pad_projector(x)

        x = x + self.pe(torch.arange(x.size(1), device=x.device))

        for block in self.blocks:
            x = block(x, x)

        outs: List[Tensor] = []
        for aggr in self.aggrs:
            out = getattr(torch, aggr)(x, dim=1)
            outs.append(out[0] if isinstance(out, tuple) else out)
        out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

        return self.fc(out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, patch_size={self.patch_size})')
