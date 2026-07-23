from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from torch_geometric.utils.mixin import CastMixin


class ExplanationType(Enum):
    model = 'model'
    phenomenon = 'phenomenon'


class MaskType(Enum):
    object = 'object'
    common_attributes = 'common_attributes'
    attributes = 'attributes'


class ModelMode(Enum):
    binary_classification = 'binary_classification'
    multiclass_classification = 'multiclass_classification'
    regression = 'regression'


class ModelTaskLevel(Enum):
    node = 'node'
    edge = 'edge'
    graph = 'graph'


class ModelReturnType(Enum):
    raw = 'raw'
    probs = 'probs'
    log_probs = 'log_probs'


class ThresholdType(Enum):
    hard = 'hard'
    topk = 'topk'
    topk_hard = 'topk_hard'


@dataclass
class ExplainerConfig(CastMixin):
    explanation_type: ExplanationType
    node_mask_type: Optional[MaskType]
    edge_mask_type: Optional[MaskType]

    def __init__(
        self,
        explanation_type: Union[ExplanationType, str],
        node_mask_type: Optional[Union[MaskType, str]] = None,
        edge_mask_type: Optional[Union[MaskType, str]] = None,
    ):
        if node_mask_type is not None:
            node_mask_type = MaskType(node_mask_type)
        if edge_mask_type is not None:
            edge_mask_type = MaskType(edge_mask_type)

        if edge_mask_type is not None and edge_mask_type != MaskType.object:
            raise ValueError(f"'edge_mask_type' needs be None or of type "
                             f"'object' (got '{edge_mask_type.value}')")

        if node_mask_type is None and edge_mask_type is None:
            raise ValueError("Either 'node_mask_type' or 'edge_mask_type' "
                             "must be provided")

        self.explanation_type = ExplanationType(explanation_type)
        self.node_mask_type = node_mask_type
        self.edge_mask_type = edge_mask_type


@dataclass
class ModelConfig(CastMixin):
    mode: ModelMode
    task_level: ModelTaskLevel
    return_type: ModelReturnType

    def __init__(
        self,
        mode: Union[ModelMode, str],
        task_level: Union[ModelTaskLevel, str],
        return_type: Optional[Union[ModelReturnType, str]] = None,
    ):
        self.mode = ModelMode(mode)
        self.task_level = ModelTaskLevel(task_level)

        if return_type is None and self.mode == ModelMode.regression:
            return_type = ModelReturnType.raw

        self.return_type = ModelReturnType(return_type)

        if (self.mode == ModelMode.regression
                and self.return_type != ModelReturnType.raw):
            raise ValueError(f"A model for regression needs to return raw "
                             f"outputs (got {self.return_type.value})")

        if (self.mode == ModelMode.binary_classification and self.return_type
                not in [ModelReturnType.raw, ModelReturnType.probs]):
            raise ValueError(
                f"A model for binary classification needs to return raw "
                f"outputs or probabilities (got {self.return_type.value})")


@dataclass
class ThresholdConfig(CastMixin):
    type: ThresholdType
    value: Union[float, int]

    def __init__(
        self,
        threshold_type: Union[ThresholdType, str],
        value: Union[float, int],
    ):
        self.type = ThresholdType(threshold_type)
        self.value = value

        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Threshold value must be a float or int "
                             f"(got {type(self.value)}).")

        if (self.type == ThresholdType.hard
                and (self.value < 0 or self.value > 1)):
            raise ValueError(f"Threshold value must be between 0 and 1 "
                             f"(got {self.value})")

        if self.type in [ThresholdType.topk, ThresholdType.topk_hard]:
            if not isinstance(self.value, int):
                raise ValueError(f"Threshold value needs to be an integer "
                                 f"(got {type(self.value)}).")
            if self.value <= 0:
                raise ValueError(f"Threshold value needs to be positive "
                                 f"(got {self.value}).")
