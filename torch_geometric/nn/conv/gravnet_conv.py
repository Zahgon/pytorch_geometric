import warnings
from typing import Optional, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.index import index2ptr
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptPairTensor  # noqa
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor


class GravNetConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 space_dimensions: int, propagate_dimensions: int, k: int,
                 num_workers: Optional[int] = None, **kwargs):
        super().__init__(aggr=['mean', 'max'], flow='source_to_target',
                         **kwargs)

        if not torch_geometric.typing.WITH_KNN:
            raise ImportError("'GravNetConv' requires 'pyg-lib>=0.6.0'")

        if num_workers is not None:
            warnings.warn(
                "'num_workers' attribute in '{self.__class__.__name__}' is "
                "deprecated and will be removed in a future release",
                stacklevel=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.lin_s = Linear(in_channels, space_dimensions)
        self.lin_h = Linear(in_channels, propagate_dimensions)

        self.lin_out1 = Linear(in_channels, out_channels, bias=False)
        self.lin_out2 = Linear(2 * propagate_dimensions, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_s.reset_parameters()
        self.lin_h.reset_parameters()
        self.lin_out1.reset_parameters()
        self.lin_out2.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        batch: Union[OptTensor, Optional[PairTensor]] = None,
    ) -> Tensor:

        is_bipartite: bool = True
        if isinstance(x, Tensor):
            x = (x, x)
            is_bipartite = False

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in 'GravNetConv'")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        h_l: Tensor = self.lin_h(x[0])

        s_l: Tensor = self.lin_s(x[0])
        s_r: Tensor = self.lin_s(x[1]) if is_bipartite else s_l

        ptr_l = None if b[0] is None else index2ptr(b[0])
        ptr_r = None if b[1] is None else index2ptr(b[1])
        edge_index = torch.ops.pyg.knn(s_l, s_r, ptr_l, ptr_r, self.k, False,
                                       1).flip([0])

        edge_weight = (s_l[edge_index[0]] - s_r[edge_index[1]]).pow(2).sum(-1)
        edge_weight = torch.exp(-10. * edge_weight)  # 10 gives a better spread

        out = self.propagate(edge_index, x=(h_l, None),
                             edge_weight=edge_weight,
                             size=(s_l.size(0), s_r.size(0)))

        return self.lin_out1(x[1]) + self.lin_out2(out)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j * edge_weight.unsqueeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, k={self.k})')
