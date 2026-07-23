from typing import Final, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.utils import scatter, segment, to_dense_batch


class Aggregation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._deterministic: Final[bool] = (
            torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled())

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            index (torch.Tensor, optional): The indices of elements for
                applying the aggregation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            ptr (torch.Tensor, optional): If given, computes the aggregation
                based on sorted inputs in CSR representation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim` after aggregation. (default: :obj:`None`)
            dim (int, optional): The dimension in which to aggregate.
                (default: :obj:`-2`)
            max_num_elements: (int, optional): The maximum number of elements
                within a single aggregation group. (default: :obj:`None`)
        """

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""

    @disable_dynamic_shapes(required_args=['dim_size'])
    def __call__(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ) -> Tensor:

        if dim >= x.dim() or dim < -x.dim():
            raise ValueError(f"Encountered invalid dimension '{dim}' of "
                             f"source tensor with {x.dim()} dimensions")

        if index is None and ptr is None:
            index = x.new_zeros(x.size(dim), dtype=torch.long)

        if ptr is not None:
            if dim_size is None:
                dim_size = ptr.numel() - 1
            elif dim_size != ptr.numel() - 1:
                raise ValueError(f"Encountered invalid 'dim_size' (got "
                                 f"'{dim_size}' but expected "
                                 f"'{ptr.numel() - 1}')")

        if index is not None and dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        try:
            return super().__call__(x, index=index, ptr=ptr, dim_size=dim_size,
                                    dim=dim, **kwargs)
        except (IndexError, RuntimeError) as e:
            if index is not None:
                if index.numel() > 0 and dim_size <= int(index.max()):
                    raise ValueError(f"Encountered invalid 'dim_size' (got "
                                     f"'{dim_size}' but expected "
                                     f">= '{int(index.max()) + 1}')") from e
            raise e

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


    def assert_index_present(self, index: Optional[Tensor]):
        if index is None:
            raise NotImplementedError(
                "Aggregation requires 'index' to be specified")

    def assert_sorted_index(self, index: Optional[Tensor]):
        if index is not None and not torch.all(index[:-1] <= index[1:]):
            raise ValueError("Can not perform aggregation since the 'index' "
                             "tensor is not sorted. Specifically, if you use "
                             "this aggregation as part of 'MessagePassing`, "
                             "ensure that 'edge_index' is sorted by "
                             "destination nodes, e.g., by calling "
                             "`data.sort(sort_by_row=False)`")

    def assert_two_dimensional_input(self, x: Tensor, dim: int):
        if x.dim() != 2:
            raise ValueError(f"Aggregation requires two-dimensional inputs "
                             f"(got '{x.dim()}')")

        if dim not in [-2, 0]:
            raise ValueError(f"Aggregation needs to perform aggregation in "
                             f"first dimension (got '{dim}')")


    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
               dim: int = -2, reduce: str = 'sum') -> Tensor:

        if ptr is not None:
            if index is None or self._deterministic:
                ptr = expand_left(ptr, dim, dims=x.dim())
                return segment(x, ptr, reduce=reduce)

        if index is None:
            raise RuntimeError("Aggregation requires 'index' to be specified")

        return scatter(x, index, dim, dim_size, reduce)

    def to_dense_batch(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        fill_value: float = 0.0,
        max_num_elements: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:

        self.assert_index_present(index)
        self.assert_sorted_index(index)
        self.assert_two_dimensional_input(x, dim)

        return to_dense_batch(
            x,
            index,
            batch_size=dim_size,
            fill_value=fill_value,
            max_num_nodes=max_num_elements,
        )




def expand_left(ptr: Tensor, dim: int, dims: int) -> Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr
