from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import SignedConv
from torch_geometric.utils import (
    coalesce,
    negative_sampling,
    structured_negative_sampling,
)


class SignedGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        lamb: float = 5,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lamb = lamb

        self.conv1 = SignedConv(in_channels, hidden_channels // 2,
                                first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                SignedConv(hidden_channels // 2, hidden_channels // 2,
                           first_aggr=False))

        self.lin = torch.nn.Linear(2 * hidden_channels, 3)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def split_edges(
        self,
        edge_index: Tensor,
        test_ratio: float = 0.2,
    ) -> Tuple[Tensor, Tensor]:
        pass

    def create_spectral_features(
        self,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        pass

    def forward(
        self,
        x: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes node embeddings :obj:`z` based on positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`.

        Args:
            x (torch.Tensor): The input node features.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index))
        return z

    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Given node embeddings :obj:`z`, classifies the link relation
        between node pairs :obj:`edge_index` to be either positive,
        negative or non-existent.

        Args:
            z (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def nll_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative nedges
        :obj:`neg_edge_index`.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1), ), 0))
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1), ), 1))
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1), ), 2))
        return nll_loss / 3.0

    def pos_embedding_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
    ) -> Tensor:
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z: Tensor, neg_edge_index: Tensor) -> Tensor:
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (torch.Tensor): The node embeddings.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes the overall objective.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_1 + loss_2)

    def test(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tuple[float, float]:
        """Evaluates node embeddings :obj:`z` on positive and negative test
        edges by computing AUC and F1 scores.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        from sklearn.metrics import f1_score, roc_auc_score

        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones(pos_p.size(0)),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        auc = roc_auc_score(y, pred)
        f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0

        return auc, f1

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, num_layers={self.num_layers})')
