from math import sqrt
from typing import Dict, Optional, Tuple, Union, overload

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import (
    ExplainerConfig,
    Explanation,
    HeteroExplanation,
    ModelConfig,
)
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import (
    clear_masks,
    set_hetero_masks,
    set_masks,
)
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from torch_geometric.typing import EdgeType, NodeType


class GNNExplainer(ExplainerAlgorithm):

    default_coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs = dict(self.default_coeffs)
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
        self.is_hetero = False

    @overload
    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        ...

    @overload
    def forward(
        self,
        model: torch.nn.Module,
        x: Dict[NodeType, Tensor],
        edge_index: Dict[EdgeType, Tensor],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> HeteroExplanation:
        ...

    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        self.is_hetero = isinstance(x, dict)
        self._train(model, x, edge_index, target=target, index=index, **kwargs)
        explanation = self._create_explanation()
        self._clean_model(model)
        return explanation

    def _create_explanation(self) -> Union[Explanation, HeteroExplanation]:
        """Create an explanation object from the current masks."""
        if self.is_hetero:
            node_mask_dict = {}
            edge_mask_dict = {}

            for node_type, mask in self.node_mask.items():
                if mask is not None:
                    node_mask_dict[node_type] = self._post_process_mask(
                        mask,
                        self.hard_node_mask[node_type],
                        apply_sigmoid=True,
                    )

            for edge_type, mask in self.edge_mask.items():
                if mask is not None:
                    edge_mask_dict[edge_type] = self._post_process_mask(
                        mask,
                        self.hard_edge_mask[edge_type],
                        apply_sigmoid=True,
                    )

            explanation = HeteroExplanation()
            explanation.set_value_dict('node_mask', node_mask_dict)
            explanation.set_value_dict('edge_mask', edge_mask_dict)

        else:
            node_mask = self._post_process_mask(
                self.node_mask,
                self.hard_node_mask,
                apply_sigmoid=True,
            )
            edge_mask = self._post_process_mask(
                self.edge_mask,
                self.hard_edge_mask,
                apply_sigmoid=True,
            )

            explanation = Explanation(node_mask=node_mask, edge_mask=edge_mask)

        return explanation

    def supports(self) -> bool:
        return True

    @overload
    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> None:
        ...

    @overload
    def _train(
        self,
        model: torch.nn.Module,
        x: Dict[NodeType, Tensor],
        edge_index: Dict[EdgeType, Tensor],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> None:
        ...

    def _train(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> None:
        self._initialize_masks(x, edge_index)

        parameters = self._collect_parameters(model, edge_index)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            y_hat = self._forward_with_masks(model, x, edge_index, **kwargs)
            y = target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            if i == 0:
                self._collect_gradients()

    def _collect_parameters(self, model, edge_index):
        """Collect parameters for optimization."""
        parameters = []

        if self.is_hetero:
            for mask in self.node_mask.values():
                if mask is not None:
                    parameters.append(mask)
            if any(v is not None for v in self.edge_mask.values()):
                set_hetero_masks(model, self.edge_mask, edge_index)
            for mask in self.edge_mask.values():
                if mask is not None:
                    parameters.append(mask)
        else:
            if self.node_mask is not None:
                parameters.append(self.node_mask)
            if self.edge_mask is not None:
                set_masks(model, self.edge_mask, edge_index,
                          apply_sigmoid=True)
                parameters.append(self.edge_mask)

        return parameters

    @overload
    def _forward_with_masks(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tensor:
        ...

    @overload
    def _forward_with_masks(
        self,
        model: torch.nn.Module,
        x: Dict[NodeType, Tensor],
        edge_index: Dict[EdgeType, Tensor],
        **kwargs,
    ) -> Tensor:
        ...

    def _forward_with_masks(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        **kwargs,
    ) -> Tensor:
        """Forward pass with masked inputs."""
        if self.is_hetero:
            h_dict = {}
            for node_type, features in x.items():
                if node_type in self.node_mask and self.node_mask[
                        node_type] is not None:
                    h_dict[node_type] = features * self.node_mask[
                        node_type].sigmoid()
                else:
                    h_dict[node_type] = features

            return model(h_dict, edge_index, **kwargs)
        else:
            h = x if self.node_mask is None else x * self.node_mask.sigmoid()

            return model(h, edge_index, **kwargs)

    def _initialize_masks(
        self,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
    ) -> None:
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        if self.is_hetero:
            self.node_mask = {}
            self.hard_node_mask = {}
            self.edge_mask = {}
            self.hard_edge_mask = {}

            for node_type, features in x.items():
                device = features.device
                N, F = features.size()
                self._initialize_node_mask(node_mask_type, node_type, N, F,
                                           device)

            for edge_type, indices in edge_index.items():
                device = indices.device
                E = indices.size(1)
                N = max(indices.max().item() + 1,
                        max(feat.size(0) for feat in x.values()))
                self._initialize_edge_mask(edge_mask_type, edge_type, E, N,
                                           device)
        else:
            device = x.device
            (N, F), E = x.size(), edge_index.size(1)

            self._initialize_homogeneous_masks(node_mask_type, edge_mask_type,
                                               N, F, E, device)

    def _initialize_node_mask(
        self,
        node_mask_type,
        node_type,
        N,
        F,
        device,
    ) -> None:
        """Initialize node mask for a specific node type."""
        std = 0.1
        if node_mask_type is None:
            self.node_mask[node_type] = None
            self.hard_node_mask[node_type] = None
        elif node_mask_type == MaskType.object:
            self.node_mask[node_type] = Parameter(
                torch.randn(N, 1, device=device) * std)
            self.hard_node_mask[node_type] = None
        elif node_mask_type == MaskType.attributes:
            self.node_mask[node_type] = Parameter(
                torch.randn(N, F, device=device) * std)
            self.hard_node_mask[node_type] = None
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask[node_type] = Parameter(
                torch.randn(1, F, device=device) * std)
            self.hard_node_mask[node_type] = None
        else:
            raise ValueError(f"Invalid node mask type: {node_mask_type}")

    def _initialize_edge_mask(self, edge_mask_type, edge_type, E, N, device):
        """Initialize edge mask for a specific edge type."""
        if edge_mask_type is None:
            self.edge_mask[edge_type] = None
            self.hard_edge_mask[edge_type] = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask[edge_type] = Parameter(
                torch.randn(E, device=device) * std)
            self.hard_edge_mask[edge_type] = None
        else:
            raise ValueError(f"Invalid edge mask type: {edge_mask_type}")

    def _initialize_homogeneous_masks(self, node_mask_type, edge_mask_type, N,
                                      F, E, device):
        """Initialize masks for homogeneous graph."""
        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            raise ValueError(f"Invalid node mask type: {node_mask_type}")

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        else:
            raise ValueError(f"Invalid edge mask type: {edge_mask_type}")

    def _collect_gradients(self) -> None:
        if self.is_hetero:
            self._collect_hetero_gradients()
        else:
            self._collect_homo_gradients()

    def _collect_hetero_gradients(self):
        """Collect gradients for heterogeneous graph."""
        for node_type, mask in self.node_mask.items():
            if mask is not None:
                if mask.grad is None:
                    raise ValueError(
                        f"Could not compute gradients for node masks of type "
                        f"'{node_type}'. Please make sure that node masks are "
                        f"used inside the model or disable it via "
                        f"`node_mask_type=None`.")

                self.hard_node_mask[node_type] = mask.grad != 0.0

        for edge_type, mask in self.edge_mask.items():
            if mask is not None:
                if mask.grad is None:
                    raise ValueError(
                        f"Could not compute gradients for edge masks of type "
                        f"'{edge_type}'. Please make sure that edge masks are "
                        f"used inside the model or disable it via "
                        f"`edge_mask_type=None`.")
                self.hard_edge_mask[edge_type] = mask.grad != 0.0

    def _collect_homo_gradients(self):
        """Collect gradients for homogeneous graph."""
        if self.node_mask is not None:
            if self.node_mask.grad is None:
                raise ValueError("Could not compute gradients for node "
                                 "features. Please make sure that node "
                                 "features are used inside the model or "
                                 "disable it via `node_mask_type=None`.")
            self.hard_node_mask = self.node_mask.grad != 0.0

        if self.edge_mask is not None:
            if self.edge_mask.grad is None:
                raise ValueError("Could not compute gradients for edges. "
                                 "Please make sure that edges are used "
                                 "via message passing inside the model or "
                                 "disable it via `edge_mask_type=None`.")
            self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        loss = self._calculate_base_loss(y_hat, y)

        if self.is_hetero:
            loss = self._apply_hetero_regularization(loss)
        else:
            loss = self._apply_homo_regularization(loss)

        return loss

    def _calculate_base_loss(self, y_hat, y):
        """Calculate base loss based on model configuration."""
        if self.model_config.mode == ModelMode.binary_classification:
            return self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            return self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            return self._loss_regression(y_hat, y)
        else:
            raise ValueError(f"Invalid model mode: {self.model_config.mode}")

    def _apply_hetero_regularization(self, loss):
        """Apply regularization for heterogeneous graph."""
        for edge_type, mask in self.edge_mask.items():
            if (mask is not None
                    and self.hard_edge_mask[edge_type] is not None):
                loss = self._add_mask_regularization(
                    loss, mask, self.hard_edge_mask[edge_type],
                    self.coeffs['edge_size'], self.coeffs['edge_reduction'],
                    self.coeffs['edge_ent'])

        for node_type, mask in self.node_mask.items():
            if (mask is not None
                    and self.hard_node_mask[node_type] is not None):
                loss = self._add_mask_regularization(
                    loss, mask, self.hard_node_mask[node_type],
                    self.coeffs['node_feat_size'],
                    self.coeffs['node_feat_reduction'],
                    self.coeffs['node_feat_ent'])

        return loss

    def _apply_homo_regularization(self, loss):
        """Apply regularization for homogeneous graph."""
        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            loss = self._add_mask_regularization(loss, self.edge_mask,
                                                 self.hard_edge_mask,
                                                 self.coeffs['edge_size'],
                                                 self.coeffs['edge_reduction'],
                                                 self.coeffs['edge_ent'])

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            loss = self._add_mask_regularization(
                loss, self.node_mask, self.hard_node_mask,
                self.coeffs['node_feat_size'],
                self.coeffs['node_feat_reduction'],
                self.coeffs['node_feat_ent'])

        return loss

    def _add_mask_regularization(self, loss, mask, hard_mask, size_coeff,
                                 reduction_name, ent_coeff):
        """Add size and entropy regularization for a mask."""
        m = mask[hard_mask].sigmoid()
        reduce_fn = getattr(torch, reduction_name)
        loss = loss + size_coeff * reduce_fn(m)
        ent = -m * torch.log(m + self.coeffs['EPS']) - (
            1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss = loss + ent_coeff * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None


class GNNExplainer_:

    coeffs = GNNExplainer.default_coeffs

    conversion_node_mask_type = {
        'feature': 'common_attributes',
        'individual_feature': 'attributes',
        'scalar': 'object',
    }

    conversion_return_type = {
        'log_prob': 'log_probs',
        'prob': 'probs',
        'raw': 'raw',
        'regression': 'raw',
    }

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        return_type: str = 'log_prob',
        feat_mask_type: str = 'feature',
        allow_edge_mask: bool = True,
        **kwargs,
    ):
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']

        explainer_config = ExplainerConfig(
            explanation_type='model',
            node_mask_type=self.conversion_node_mask_type[feat_mask_type],
            edge_mask_type=MaskType.object if allow_edge_mask else None,
        )
        model_config = ModelConfig(
            mode='regression'
            if return_type == 'regression' else 'multiclass_classification',
            task_level=ModelTaskLevel.node,
            return_type=self.conversion_return_type[return_type],
        )

        self.model = model
        self._explainer = GNNExplainer(epochs=epochs, lr=lr, **kwargs)
        self._explainer.connect(explainer_config, model_config)

    @torch.no_grad()
    def get_initial_prediction(self, *args, **kwargs) -> Tensor:
        pass

    def explain_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        pass

    def explain_node(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        pass

    def _convert_output(self, explanation, edge_index, index=None, x=None):
        pass
