import torch
from torch import Tensor


class MaskLabel(torch.nn.Module):
    def __init__(self, num_classes: int, out_channels: int,
                 method: str = "add"):
        super().__init__()

        self.method = method
        if method not in ["add", "concat"]:
            raise ValueError(
                f"'method' must be either 'add' or 'concat' (got '{method}')")

        self.emb = torch.nn.Embedding(num_classes, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.emb.reset_parameters()

    def forward(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """"""  # noqa: D419
        if self.method == "concat":
            out = x.new_zeros(y.size(0), self.emb.weight.size(-1))
            out[mask] = self.emb(y[mask])
            return torch.cat([x, out], dim=-1)
        else:
            x = torch.clone(x)
            x[mask] += self.emb(y[mask])
            return x

    @staticmethod
    def ratio_mask(mask: Tensor, ratio: float):
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
