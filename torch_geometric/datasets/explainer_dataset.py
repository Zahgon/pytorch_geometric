from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.datasets.motif_generator import MotifGenerator
from torch_geometric.explain import Explanation


class ExplainerDataset(InMemoryDataset):
    def __init__(
        self,
        graph_generator: Union[GraphGenerator, str],
        motif_generator: Union[MotifGenerator, str],
        num_motifs: int,
        num_graphs: int = 1,
        graph_generator_kwargs: Optional[Dict[str, Any]] = None,
        motif_generator_kwargs: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root=None, transform=transform)

        if num_motifs <= 0:
            raise ValueError(f"At least one motif needs to be attached to the "
                             f"graph (got {num_motifs})")

        self.graph_generator = GraphGenerator.resolve(
            graph_generator,
            **(graph_generator_kwargs or {}),
        )
        self.motif_generator = MotifGenerator.resolve(
            motif_generator,
            **(motif_generator_kwargs or {}),
        )
        self.num_motifs = num_motifs

        data_list = [self.get_graph() for _ in range(num_graphs)]
        self.data, self.slices = self.collate(data_list)

    def get_graph(self) -> Explanation:
        data = self.graph_generator()
        assert data.num_nodes is not None
        assert data.edge_index is not None

        edge_indices = [data.edge_index]
        num_nodes = data.num_nodes
        node_masks = [torch.zeros(data.num_nodes)]
        edge_masks = [torch.zeros(data.num_edges)]
        ys = [torch.zeros(num_nodes, dtype=torch.long)]

        connecting_nodes = torch.randperm(num_nodes)[:self.num_motifs]
        for i in connecting_nodes.tolist():
            motif = self.motif_generator()
            assert motif.num_nodes is not None
            assert motif.edge_index is not None

            edge_indices.append(motif.edge_index + num_nodes)
            node_masks.append(torch.ones(motif.num_nodes))
            edge_masks.append(torch.ones(motif.num_edges))

            j = int(torch.randint(0, motif.num_nodes, (1, ))) + num_nodes
            edge_indices.append(torch.tensor([[i, j], [j, i]]))
            edge_masks.append(torch.zeros(2))

            if isinstance(motif.y, Tensor):
                ys.append(motif.y + 1 if motif.y.min() == 0 else motif.y)
            else:
                ys.append(torch.ones(motif.num_nodes, dtype=torch.long))

            num_nodes += motif.num_nodes

        return Explanation(
            edge_index=torch.cat(edge_indices, dim=1),
            y=torch.cat(ys, dim=0),
            edge_mask=torch.cat(edge_masks, dim=0),
            node_mask=torch.cat(node_masks, dim=0),
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'graph_generator={self.graph_generator}, '
                f'motif_generator={self.motif_generator}, '
                f'num_motifs={self.num_motifs})')
