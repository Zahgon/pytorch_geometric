from typing import Tuple, Union

from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor


class LEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin1 = Linear(in_channels[0], out_channels, bias=bias)
        self.lin2 = Linear(in_channels[1], out_channels, bias=False)
        self.lin3 = Linear(in_channels[1], out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        a = self.lin1(x[0])
        b = self.lin2(x[1])

        out = self.propagate(edge_index, a=a, b=b, edge_weight=edge_weight)

        return out + self.lin3(x[1])

    def message(self, a_j: Tensor, b_i: Tensor,
                edge_weight: OptTensor) -> Tensor:
        out = a_j - b_i
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
