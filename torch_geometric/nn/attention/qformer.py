from typing import Callable

import torch


class QFormer(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float = 0.0,
            activation: Callable = torch.nn.ReLU(),
    ) -> None:

        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
        )
        self.project = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Input sequence to the encoder layer.
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, sequence length :math:`N`,
                and feature dimension :math:`F`.
        """
        x = self.layer_norm(x)
        x = self.encoder(x)
        out = self.project(x)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_heads={self.num_heads}, '
                f'num_layers={self.num_layers})')
