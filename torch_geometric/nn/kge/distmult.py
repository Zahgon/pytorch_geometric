import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.kge import KGEModel


class DistMult(KGEModel):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.margin = margin

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        return (head * rel * tail).sum(dim=-1)

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )
