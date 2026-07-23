from typing import Callable, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class NNConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, nn: Callable, aggr: str = 'add',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]

        if root_weight:
            self.lin = Linear(in_channels[1], out_channels, bias=False,
                              weight_initializer='uniform')

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        if self.root_weight:
            self.lin.reset_parameters()
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root_weight:
            out = out + self.lin(x_r)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        weight = self.nn(edge_attr)
        weight = weight.view(-1, self.in_channels_l, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr}, nn={self.nn})')
