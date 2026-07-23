from typing import Callable, Optional, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.index import index2ptr
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor


class EdgeConv(MessagePassing):
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        return self.propagate(edge_index, x=x)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class DynamicEdgeConv(MessagePassing):
    def __init__(self, nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)

        if not torch_geometric.typing.WITH_KNN:
            raise ImportError("'DynamicEdgeConv' requires 'pyg-lib>=0.6.0'")

        self.nn = nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        batch: Union[OptTensor, Optional[PairTensor]] = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in DynamicEdgeConv")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        ptr_l = None if b[0] is None else index2ptr(b[0])
        ptr_r = None if b[1] is None else index2ptr(b[1])
        edge_index = torch.ops.pyg.knn(x[0], x[1], ptr_l, ptr_r, self.k, False,
                                       1).flip([0])

        return self.propagate(edge_index, x=x)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn}, k={self.k})'
