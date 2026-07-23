from typing import Optional

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter


class PairNorm(torch.nn.Module):
    def __init__(self, scale: float = 1., scale_individually: bool = False,
                 eps: float = 1e-5):
        super().__init__()

        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps

    def forward(self, x: Tensor, batch: OptTensor = None,
                batch_size: Optional[int] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        """
        scale = self.scale

        if batch is None:
            x = x - x.mean(dim=0, keepdim=True)

            if not self.scale_individually:
                return scale * x / (self.eps + x.pow(2).sum(-1).mean()).sqrt()
            else:
                return scale * x / (self.eps + x.norm(2, -1, keepdim=True))

        else:
            mean = scatter(x, batch, dim=0, dim_size=batch_size, reduce='mean')
            x = x - mean.index_select(0, batch)

            if not self.scale_individually:
                return scale * x / torch.sqrt(self.eps + scatter(
                    x.pow(2).sum(-1, keepdim=True), batch, dim=0,
                    dim_size=batch_size, reduce='mean').index_select(0, batch))
            else:
                return scale * x / (self.eps + x.norm(2, -1, keepdim=True))

    def __repr__(self):
        return f'{self.__class__.__name__}()'
