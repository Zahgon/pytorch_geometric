import inspect
import warnings
from typing import Any, Callable, Dict, Final, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import NoneType


class MLP(torch.nn.Module):
    supports_norm_batch: Final[bool]

    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Union[float, List[float]] = 0.,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        plain_last: bool = True,
        bias: Union[bool, List[bool]] = True,
        **kwargs,
    ):
        super().__init__()

        act_first = act_first or kwargs.get("relu_first", False)
        batch_norm = kwargs.get("batch_norm", None)
        if batch_norm is not None and isinstance(batch_norm, bool):
            warnings.warn(
                "Argument `batch_norm` is deprecated, "
                "please use `norm` to specify normalization layer.",
                stacklevel=2)
            norm = 'batch_norm' if batch_norm else None
            batch_norm_kwargs = kwargs.get("batch_norm_kwargs", None)
            norm_kwargs = batch_norm_kwargs or {}

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_channels is None:
                raise ValueError(f"Argument `hidden_channels` must be given "
                                 f"for `num_layers={num_layers}`")
            if out_channels is None:
                raise ValueError("Argument `out_channels` must be given")

            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
            if plain_last:
                dropout[-1] = 0.
        if len(dropout) != len(channel_list) - 1:
            raise ValueError(
                f"Number of dropout values provided ({len(dropout)} does not "
                f"match the number of layers specified "
                f"({len(channel_list)-1})")
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list)-1})")

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias)
        for in_channels, out_channels, _bias in iterator:
            self.lins.append(Linear(in_channels, out_channels, bias=_bias))

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if norm is not None:
                norm_layer = normalization_resolver(
                    norm,
                    hidden_channels,
                    **(norm_kwargs or {}),
                )
            else:
                norm_layer = Identity()
            self.norms.append(norm_layer)

        self.supports_norm_batch = False
        if len(self.norms) > 0 and hasattr(self.norms[0], 'forward'):
            norm_params = inspect.signature(self.norms[0].forward).parameters
            self.supports_norm_batch = 'batch' in norm_params

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        pass

    @property
    def out_channels(self) -> int:
        pass

    @property
    def num_layers(self) -> int:
        pass

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        return_emb: NoneType = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            return_emb (bool, optional): If set to :obj:`True`, will
                additionally return the embeddings before execution of the
                final output layer. (default: :obj:`False`)
        """
        emb: Optional[Tensor] = None

        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.supports_norm_batch:
                x = norm(x, batch, batch_size)
            else:
                x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            if isinstance(return_emb, bool) and return_emb is True:
                emb = x

        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return (x, emb) if isinstance(return_emb, bool) else x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'
