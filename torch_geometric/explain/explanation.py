import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data.data import Data, warn_or_raise
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.explain.config import ThresholdConfig, ThresholdType
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.visualization import (
    visualize_graph,
    visualize_hetero_graph,
)


class ExplanationMixin:
    @property
    def available_explanations(self) -> List[str]:
        pass

    def validate_masks(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the :class:`Explanation` masks."""
        status = True

        for store in self.node_stores:
            if 'node_mask' not in store:
                continue

            if store.node_mask.dim() != 2:
                status = False
                warn_or_raise(
                    f"Expected a 'node_mask' with two dimensions (got "
                    f"{store.node_mask.dim()} dimensions)", raise_on_error)

            if store.node_mask.size(0) not in {1, store.num_nodes}:
                status = False
                warn_or_raise(
                    f"Expected a 'node_mask' with {store.num_nodes} nodes "
                    f"(got {store.node_mask.size(0)} nodes)", raise_on_error)

            if 'x' in store:
                num_features = store.x.size(-1)
            else:
                num_features = store.node_mask.size(-1)

            if store.node_mask.size(1) not in {1, num_features}:
                status = False
                warn_or_raise(
                    f"Expected a 'node_mask' with {num_features} features ("
                    f"got {store.node_mask.size(1)} features)", raise_on_error)

        for store in self.edge_stores:
            if 'edge_mask' not in store:
                continue

            if store.edge_mask.dim() != 1:
                status = False
                warn_or_raise(
                    f"Expected an 'edge_mask' with one dimension (got "
                    f"{store.edge_mask.dim()} dimensions)", raise_on_error)

            if store.edge_mask.size(0) != store.num_edges:
                status = False
                warn_or_raise(
                    f"Expected an 'edge_mask' with {store.num_edges} edges "
                    f"(got {store.edge_mask.size(0)} edges)", raise_on_error)

        return status

    def _threshold_mask(
        self,
        mask: Optional[Tensor],
        threshold_config: ThresholdConfig,
    ) -> Optional[Tensor]:
        pass

    def threshold(
        self,
        *args,
        **kwargs,
    ) -> Union['Explanation', 'HeteroExplanation']:
        pass


class Explanation(Data, ExplanationMixin):
    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the :class:`Explanation` object."""
        status = super().validate(raise_on_error)
        status &= self.validate_masks(raise_on_error)
        return status

    def get_explanation_subgraph(self) -> 'Explanation':
        r"""Returns the induced subgraph, in which all nodes and edges with
        zero attribution are masked out.
        """
        node_mask = self.get('node_mask')
        if node_mask is not None:
            node_mask = node_mask.sum(dim=-1) > 0
        edge_mask = self.get('edge_mask')
        if edge_mask is not None:
            edge_mask = edge_mask > 0
        return self._apply_masks(node_mask, edge_mask)

    def get_complement_subgraph(self) -> 'Explanation':
        pass

    def _apply_masks(
        self,
        node_mask: Optional[Tensor] = None,
        edge_mask: Optional[Tensor] = None,
    ) -> 'Explanation':
        out = copy.copy(self)

        if edge_mask is not None:
            for key, value in self.items():
                if key == 'edge_index':
                    out.edge_index = value[:, edge_mask]
                elif self.is_edge_attr(key):
                    out[key] = value[edge_mask]

        if node_mask is not None:
            out = out.subgraph(node_mask)

        return out

    def visualize_feature_importance(
        self,
        path: Optional[str] = None,
        feat_labels: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ):
        pass

    def visualize_graph(
        self,
        path: Optional[str] = None,
        backend: Optional[str] = None,
        node_labels: Optional[List[str]] = None,
    ) -> None:
        r"""Visualizes the explanation graph with edge opacity corresponding to
        edge importance.

        Args:
            path (str, optional): The path to where the plot is saved.
                If set to :obj:`None`, will visualize the plot on-the-fly.
                (default: :obj:`None`)
            backend (str, optional): The graph drawing backend to use for
                visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
                If set to :obj:`None`, will use the most appropriate
                visualization backend based on available system packages.
                (default: :obj:`None`)
            node_labels (list[str], optional): The labels/IDs of nodes.
                (default: :obj:`None`)
        """
        edge_mask = self.get('edge_mask')
        if edge_mask is None:
            raise ValueError(f"The attribute 'edge_mask' is not available "
                             f"in '{self.__class__.__name__}' "
                             f"(got {self.available_explanations})")
        visualize_graph(self.edge_index, edge_mask, path, backend, node_labels)


class HeteroExplanation(HeteroData, ExplanationMixin):
    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the :class:`Explanation` object."""
        status = super().validate(raise_on_error)
        status &= self.validate_masks(raise_on_error)
        return status

    def get_explanation_subgraph(self) -> 'HeteroExplanation':
        r"""Returns the induced subgraph, in which all nodes and edges with
        zero attribution are masked out.
        """
        return self._apply_masks(
            node_mask_dict={
                key: mask.sum(dim=-1) > 0
                for key, mask in self.collect('node_mask', True).items()
            },
            edge_mask_dict={
                key: mask > 0
                for key, mask in self.collect('edge_mask', True).items()
            },
        )

    def get_complement_subgraph(self) -> 'HeteroExplanation':
        pass

    def _apply_masks(
        self,
        node_mask_dict: Dict[NodeType, Tensor],
        edge_mask_dict: Dict[EdgeType, Tensor],
    ) -> 'HeteroExplanation':
        out = copy.copy(self)

        for edge_type, edge_mask in edge_mask_dict.items():
            for key, value in self[edge_type].items():
                if key == 'edge_index':
                    out[edge_type].edge_index = value[:, edge_mask]
                elif self[edge_type].is_edge_attr(key):
                    out[edge_type][key] = value[edge_mask]

        return out.subgraph(node_mask_dict)

    def visualize_feature_importance(
        self,
        path: Optional[str] = None,
        feat_labels: Optional[Dict[NodeType, List[str]]] = None,
        top_k: Optional[int] = None,
    ):
        pass

    def visualize_graph(
            self,
            path: Optional[str] = None,
            node_labels: Optional[Dict[NodeType, List[str]]] = None,
            node_size_range: Tuple[float, float] = (50, 500),
            node_opacity_range: Tuple[float, float] = (0.2, 1.0),
            edge_width_range: Tuple[float, float] = (0.1, 2.0),
            edge_opacity_range: Tuple[float, float] = (0.2, 1.0),
    ) -> None:
        r"""Visualizes the explanation subgraph using networkx, with edge
        opacity corresponding to edge importance and node colors
        corresponding to node types.

        Args:
            path (str, optional): The path to where the plot is saved.
                If set to :obj:`None`, will visualize the plot on-the-fly.
                (default: :obj:`None`)
            node_labels (Dict[NodeType, List[str]], optional): The display
                names of nodes for each node type that will be shown in the
                visualization. (default: :obj:`None`)
            node_size_range (Tuple[float, float], optional): The minimum and
                maximum node size in the visualization.
                (default: :obj:`(50, 500)`)
            node_opacity_range (Tuple[float, float], optional): The minimum and
                maximum node opacity in the visualization.
                (default: :obj:`(0.2, 1.0)`)
            edge_width_range (Tuple[float, float], optional): The minimum and
                maximum edge width in the visualization.
                (default: :obj:`(0.1, 2.0)`)
            edge_opacity_range (Tuple[float, float], optional): The minimum and
                maximum edge opacity in the visualization.
                (default: :obj:`(0.2, 1.0)`)
        """
        if node_labels is not None:
            for node_type, labels in node_labels.items():
                if node_type not in self.node_types:
                    raise ValueError(
                        f"Node type '{node_type}' in node_labels "
                        f"does not exist in the explanation graph")
                if len(labels) != self[node_type].num_nodes:
                    raise ValueError(f"Number of labels for node type "
                                     f"'{node_type}' (got {len(labels)}) does "
                                     f"not match the number of nodes "
                                     f"(got {self[node_type].num_nodes})")
        subgraph = self.get_explanation_subgraph()

        edge_index_dict = {}
        edge_weight_dict = {}
        for edge_type in subgraph.edge_types:
            if edge_type[0] == 'x' or edge_type[-1] == 'x':  # Skip edges
                continue
            edge_index_dict[edge_type] = subgraph[edge_type].edge_index
            edge_weight_dict[edge_type] = subgraph[edge_type].get(
                'edge_mask',
                torch.ones(subgraph[edge_type].edge_index.size(1)))

        node_weight_dict = {}
        for node_type in subgraph.node_types:
            if node_type == 'x':  # Skip the global store
                continue
            node_weight_dict[node_type] = subgraph[node_type] \
                .get('node_mask',
                     torch.ones(subgraph[node_type].num_nodes)).squeeze(-1)

        visualize_hetero_graph(
            edge_index_dict=edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            path=path,
            node_labels_dict=node_labels,
            node_weight_dict=node_weight_dict,
            node_size_range=node_size_range,
            node_opacity_range=node_opacity_range,
            edge_width_range=edge_width_range,
            edge_opacity_range=edge_opacity_range,
        )


def _visualize_score(
    score: torch.Tensor,
    labels: List[str],
    path: Optional[str] = None,
    top_k: Optional[int] = None,
):
    pass
