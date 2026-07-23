import copy
from typing import Callable, Optional, Tuple, Union

from torch import Tensor
from torch.nn import ModuleList, ReLU

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)


class FiLMConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            num_relations: int = 1,
            nn: Optional[Callable] = None,
            act: Optional[Callable] = ReLU(),
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = max(num_relations, 1)
        self.act = act

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lins = ModuleList()
        self.films = ModuleList()
        for _ in range(num_relations):
            self.lins.append(Linear(in_channels[0], out_channels, bias=False))
            if nn is None:
                film = Linear(in_channels[1], 2 * out_channels)
            else:
                film = copy.deepcopy(nn)
            self.films.append(film)

        self.lin_skip = Linear(in_channels[1], self.out_channels, bias=False)
        if nn is None:
            self.film_skip = Linear(in_channels[1], 2 * self.out_channels,
                                    bias=False)
        else:
            self.film_skip = copy.deepcopy(nn)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin, film in zip(self.lins, self.films):
            lin.reset_parameters()
            reset(film)
        self.lin_skip.reset_parameters()
        reset(self.film_skip)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_type: OptTensor = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        beta, gamma = self.film_skip(x[1]).split(self.out_channels, dim=-1)
        out = gamma * self.lin_skip(x[1]) + beta
        if self.act is not None:
            out = self.act(out)

        if self.num_relations <= 1:
            beta, gamma = self.films[0](x[1]).split(self.out_channels, dim=-1)
            out = out + self.propagate(edge_index, x=self.lins[0](x[0]),
                                       beta=beta, gamma=gamma)
        else:
            for i, (lin, film) in enumerate(zip(self.lins, self.films)):
                beta, gamma = film(x[1]).split(self.out_channels, dim=-1)
                if isinstance(edge_index, SparseTensor):
                    _edge_type = edge_index.storage.value()
                    assert _edge_type is not None
                    mask = _edge_type == i
                    adj_t = torch_sparse.masked_select_nnz(
                        edge_index, mask, layout='coo')
                    out = out + self.propagate(adj_t, x=lin(x[0]), beta=beta,
                                               gamma=gamma)
                else:
                    assert edge_type is not None
                    mask = edge_type == i
                    out = out + self.propagate(edge_index[:, mask], x=lin(
                        x[0]), beta=beta, gamma=gamma)

        return out

    def message(self, x_j: Tensor, beta_i: Tensor, gamma_i: Tensor) -> Tensor:
        out = gamma_i * x_j + beta_i
        if self.act is not None:
            out = self.act(out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
