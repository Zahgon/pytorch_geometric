import copy
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import Adj


class InvertibleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn: torch.nn.Module, fn_inverse: torch.nn.Module,
                num_bwd_passes: int, num_inputs: int, *args):
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.weights = args[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.num_inputs = num_inputs
        inputs = args[:num_inputs]
        ctx.input_requires_grad = []

        with torch.no_grad():  # Make a detached copy which shares the storage:
            x = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    x.append(element.detach())
                    ctx.input_requires_grad.append(element.requires_grad)
                else:
                    x.append(element)
                    ctx.input_requires_grad.append(None)
            outputs = ctx.fn(*x)

        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        detached_outputs = tuple(element.detach_() for element in outputs)

        if torch_geometric.typing.WITH_PT20:
            inputs[0].untyped_storage().resize_(0)
        else:  # pragma: no cover
            inputs[0].storage().resize_(0)

        ctx.inputs = [inputs] * num_bwd_passes
        ctx.outputs = [detached_outputs] * num_bwd_passes

        return detached_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(ctx.outputs) == 0:
            raise RuntimeError(
                f"Trying to perform a backward pass on the "
                f"'InvertibleFunction' for more than '{ctx.num_bwd_passes}' "
                f"times. Try raising 'num_bwd_passes'.")

        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        with torch.no_grad():
            inputs_inverted = ctx.fn_inverse(*(outputs + inputs[1:]))
            if len(ctx.outputs) == 0:  # Clear memory from outputs:
                for element in outputs:
                    if torch_geometric.typing.WITH_PT20:
                        element.untyped_storage().resize_(0)
                    else:  # pragma: no cover
                        element.storage().resize_(0)

            if not isinstance(inputs_inverted, tuple):
                inputs_inverted = (inputs_inverted, )

            for elem_orig, elem_inv in zip(inputs, inputs_inverted):
                if torch_geometric.typing.WITH_PT20:
                    elem_orig.untyped_storage().resize_(
                        int(np.prod(elem_orig.size())) *
                        elem_orig.element_size())
                else:  # pragma: no cover
                    elem_orig.storage().resize_(int(np.prod(elem_orig.size())))
                elem_orig.set_(elem_inv)

        with torch.set_grad_enabled(True):
            detached_inputs = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    detached_inputs.append(element.detach())
                else:
                    detached_inputs.append(element)
            detached_inputs = tuple(detached_inputs)
            for x, req_grad in zip(detached_inputs, ctx.input_requires_grad):
                if isinstance(x, torch.Tensor):
                    x.requires_grad = req_grad
            tmp_output = ctx.fn(*detached_inputs)

        if not isinstance(tmp_output, tuple):
            tmp_output = (tmp_output, )

        filtered_detached_inputs = tuple(
            filter(
                lambda x: x.requires_grad
                if isinstance(x, torch.Tensor) else False,
                detached_inputs,
            ))
        gradients = torch.autograd.grad(
            outputs=tmp_output,
            inputs=filtered_detached_inputs + ctx.weights,
            grad_outputs=grad_outputs,
        )

        input_gradients = []
        i = 0
        for rg in ctx.input_requires_grad:
            if rg:
                input_gradients.append(gradients[i])
                i += 1
            else:
                input_gradients.append(None)

        gradients = tuple(input_gradients) + gradients[-len(ctx.weights):]

        return (None, None, None, None) + gradients


class InvertibleModule(torch.nn.Module):
    def __init__(self, disable: bool = False, num_bwd_passes: int = 1):
        super().__init__()
        self.disable = disable
        self.num_bwd_passes = num_bwd_passes

    def forward(self, *args):
        """"""  # noqa: D419
        return self._fn_apply(args, self._forward, self._inverse)

    def inverse(self, *args):
        return self._fn_apply(args, self._inverse, self._forward)

    def _forward(self):
        raise NotImplementedError

    def _inverse(self):
        raise NotImplementedError

    def _fn_apply(self, args, fn, fn_inverse):
        if not self.disable:
            out = InvertibleFunction.apply(
                fn,
                fn_inverse,
                self.num_bwd_passes,
                len(args),
                *args,
                *tuple(p for p in self.parameters() if p.requires_grad),
            )
        else:
            out = fn(*args)

        if isinstance(out, tuple) and len(out) == 1:
            return out[0]

        return out


class GroupAddRev(InvertibleModule):
    def __init__(
        self,
        conv: Union[torch.nn.Module, torch.nn.ModuleList],
        split_dim: int = -1,
        num_groups: Optional[int] = None,
        disable: bool = False,
        num_bwd_passes: int = 1,
    ):
        super().__init__(disable, num_bwd_passes)
        self.split_dim = split_dim

        if isinstance(conv, torch.nn.ModuleList):
            self.convs = conv
        else:
            assert num_groups is not None, "Please specific 'num_groups'"
            self.convs = torch.nn.ModuleList([conv])
            for _ in range(num_groups - 1):
                conv = copy.deepcopy(self.convs[0])
                if hasattr(conv, 'reset_parameters'):
                    conv.reset_parameters()
                self.convs.append(conv)

        if len(self.convs) < 2:
            raise ValueError(f"The number of groups should not be smaller "
                             f"than '2' (got '{self.num_groups}'))")

    @property
    def num_groups(self) -> int:
        pass

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()

    def _forward(self, x: Tensor, edge_index: Adj, *args):
        channels = x.size(self.split_dim)
        xs = self._chunk(x, channels)
        args = list(zip(*[self._chunk(arg, channels) for arg in args]))
        args = [[]] * self.num_groups if len(args) == 0 else args

        ys = []
        y_in = sum(xs[1:])
        for i in range(self.num_groups):
            y_in = xs[i] + self.convs[i](y_in, edge_index, *args[i])
            ys.append(y_in)
        return torch.cat(ys, dim=self.split_dim)

    def _inverse(self, y: Tensor, edge_index: Adj, *args):
        pass

    def _chunk(self, x: Any, channels: int) -> List[Any]:
        if not isinstance(x, Tensor):
            return [x] * self.num_groups

        try:
            if x.size(self.split_dim) != channels:
                return [x] * self.num_groups
        except IndexError:
            return [x] * self.num_groups

        return torch.chunk(x, self.num_groups, dim=self.split_dim)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.convs[0]}, '
                f'num_groups={self.num_groups})')
