from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.utils import cumsum, scatter

try:
    import torchmetrics  # noqa
    WITH_TORCHMETRICS = True
    BaseMetric = torchmetrics.Metric
except Exception:
    WITH_TORCHMETRICS = False
    BaseMetric = torch.nn.Module  # type: ignore


@dataclass(repr=False)
class LinkPredMetricData:
    pred_index_mat: Tensor
    edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]]
    edge_label_weight: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.edge_label_weight is not None:
            pos_mask = self.edge_label_weight > 0
            self.edge_label_weight = self.edge_label_weight[pos_mask]
            if isinstance(self.edge_label_index, Tensor):
                self.edge_label_index = self.edge_label_index[:, pos_mask]
            else:
                self.edge_label_index = (
                    self.edge_label_index[0][pos_mask],
                    self.edge_label_index[1][pos_mask],
                )

    @property
    def pred_rel_mat(self) -> Tensor:
        pass

    @property
    def label_count(self) -> Tensor:
        pass

    @property
    def label_weight_sum(self) -> Tensor:
        pass

    @property
    def edge_label_weight_pos(self) -> Optional[Tensor]:
        pass


class _LinkPredMetric(BaseMetric):
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: Optional[bool] = None

    def __init__(self, k: int) -> None:
        super().__init__()

        if k <= 0:
            raise ValueError(f"'k' needs to be a positive integer in "
                             f"'{self.__class__.__name__}' (got {k})")

        self.k = k

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        r"""Updates the state variables based on the current mini-batch
        prediction.

        :meth:`update` can be repeated multiple times to accumulate the results
        of successive predictions, *e.g.*, inside a mini-batch training or
        evaluation loop.

        Args:
            pred_index_mat (torch.Tensor): The top-:math:`k` predictions of
                every example in the mini-batch with shape
                :obj:`[batch_size, k]`.
            edge_label_index (torch.Tensor): The ground-truth indices for every
                example in the mini-batch, given in COO format of shape
                :obj:`[2, num_ground_truth_indices]`.
            edge_label_weight (torch.Tensor, optional): The weight of the
                ground-truth indices for every example in the mini-batch of
                shape :obj:`[num_ground_truth_indices]`. If given, needs to be
                a vector of positive values. Required for weighted metrics,
                ignored otherwise. (default: :obj:`None`)
        """
        raise NotImplementedError

    def compute(self) -> Tensor:
        r"""Computes the final metric value."""
        raise NotImplementedError

    def reset(self) -> None:
        r"""Resets metric state variables to their default value."""
        if WITH_TORCHMETRICS:
            super().reset()
        else:
            self._reset()

    def _reset(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'


class LinkPredMetric(_LinkPredMetric):
    weighted: bool

    def __init__(self, k: int) -> None:
        super().__init__(k)

        self.accum: Tensor
        self.total: Tensor

        if WITH_TORCHMETRICS:
            self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.register_buffer('accum', torch.tensor(0.), persistent=False)
            self.register_buffer('total', torch.tensor(0), persistent=False)

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        if self.weighted and edge_label_weight is None:
            raise ValueError(f"'edge_label_weight' is a required argument for "
                             f"weighted '{self.__class__.__name__}' metrics")
        if not self.weighted:
            edge_label_weight = None

        data = LinkPredMetricData(
            pred_index_mat=pred_index_mat,
            edge_label_index=edge_label_index,
            edge_label_weight=edge_label_weight,
        )
        self._update(data)

    def _update(self, data: LinkPredMetricData) -> None:
        metric = self._compute(data)

        self.accum += metric.sum()
        self.total += (data.label_count > 0).sum()

    def compute(self) -> Tensor:
        pass

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        r"""Computes the specific metric.
        To be implemented separately for each metric class.

        Args:
            data (LinkPredMetricData): The mini-batch data for computing a link
                prediction metric per example.
        """
        raise NotImplementedError

    def _reset(self) -> None:
        self.accum.zero_()
        self.total.zero_()

    def __repr__(self) -> str:
        weighted_repr = ', weighted=True' if self.weighted else ''
        return f'{self.__class__.__name__}(k={self.k}{weighted_repr})'


class LinkPredMetricCollection(torch.nn.ModuleDict):
    def __init__(
        self,
        metrics: Union[
            List[LinkPredMetric],
            Dict[str, LinkPredMetric],
        ],
    ) -> None:
        super().__init__()

        if isinstance(metrics, (list, tuple)):
            metrics = {
                (f'{"Weighted" if getattr(metric, "weighted", False) else ""}'
                 f'{metric.__class__.__name__}@{metric.k}'):
                metric
                for metric in metrics
            }
        assert len(metrics) > 0
        assert isinstance(metrics, dict)

        for name, metric in metrics.items():
            assert isinstance(metric, _LinkPredMetric)
            self[name] = metric

    @property
    def max_k(self) -> int:
        pass

    @property
    def weighted(self) -> bool:
        pass

    def update(  # type: ignore
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        r"""Updates the state variables based on the current mini-batch
        prediction.

        :meth:`update` can be repeated multiple times to accumulate the results
        of successive predictions, *e.g.*, inside a mini-batch training or
        evaluation loop.

        Args:
            pred_index_mat (torch.Tensor): The top-:math:`k` predictions of
                every example in the mini-batch with shape
                :obj:`[batch_size, k]`.
            edge_label_index (torch.Tensor): The ground-truth indices for every
                example in the mini-batch, given in COO format of shape
                :obj:`[2, num_ground_truth_indices]`.
            edge_label_weight (torch.Tensor, optional): The weight of the
                ground-truth indices for every example in the mini-batch of
                shape :obj:`[num_ground_truth_indices]`. If given, needs to be
                a vector of positive values. Required for weighted metrics,
                ignored otherwise. (default: :obj:`None`)
        """
        if self.weighted and edge_label_weight is None:
            raise ValueError(f"'edge_label_weight' is a required argument for "
                             f"weighted '{self.__class__.__name__}' metrics")

        data = LinkPredMetricData(  # Share metric data across metrics.
            pred_index_mat=pred_index_mat,
            edge_label_index=edge_label_index,
            edge_label_weight=edge_label_weight,
        )

        for metric in self.values():
            if isinstance(metric, LinkPredMetric) and metric.weighted:
                metric._update(data)
                if WITH_TORCHMETRICS:
                    metric._update_count += 1

        data.edge_label_weight = None
        if hasattr(data, '_pred_rel_mat'):
            data._pred_rel_mat = data._pred_rel_mat != 0.0
        if hasattr(data, '_label_weight_sum'):
            del data._label_weight_sum
        if hasattr(data, '_edge_label_weight_pos'):
            del data._edge_label_weight_pos

        for metric in self.values():
            if isinstance(metric, LinkPredMetric) and not metric.weighted:
                metric._update(data)
                if WITH_TORCHMETRICS:
                    metric._update_count += 1

        for metric in self.values():
            if not isinstance(metric, LinkPredMetric):
                metric.update(  # type: ignore[operator]
                    pred_index_mat,
                    edge_label_index,
                    edge_label_weight,
                )

    def compute(self) -> Dict[str, Tensor]:
        pass

    def reset(self) -> None:
        r"""Reset metric state variables to their default value."""
        for metric in self.values():
            metric.reset()  # type: ignore[operator]

    def __repr__(self) -> str:
        names = [f'  {name}: {metric},\n' for name, metric in self.items()]
        return f'{self.__class__.__name__}([\n{"".join(names)}])'


class LinkPredPrecision(LinkPredMetric):
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        return pred_rel_mat.sum(dim=-1) / self.k


class LinkPredRecall(LinkPredMetric):
    higher_is_better: bool = True

    def __init__(self, k: int, weighted: bool = False):
        super().__init__(k=k)
        self.weighted = weighted

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        return pred_rel_mat.sum(dim=-1) / data.label_weight_sum.clamp(min=1e-7)


class LinkPredF1(LinkPredMetric):
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        isin_count = pred_rel_mat.sum(dim=-1)
        precision = isin_count / self.k
        recall = isin_count / data.label_count.clamp(min=1e-7)
        return 2 * precision * recall / (precision + recall).clamp(min=1e-7)


class LinkPredMAP(LinkPredMetric):
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        device = pred_rel_mat.device
        arange = torch.arange(1, pred_rel_mat.size(1) + 1, device=device)
        cum_precision = pred_rel_mat.cumsum(dim=1) / arange
        return ((cum_precision * pred_rel_mat).sum(dim=-1) /
                data.label_count.clamp(min=1e-7, max=self.k))


class LinkPredNDCG(LinkPredMetric):
    higher_is_better: bool = True

    def __init__(self, k: int, weighted: bool = False):
        super().__init__(k=k)
        self.weighted = weighted

        dtype = torch.get_default_dtype()
        discount = torch.arange(2, k + 2, dtype=dtype).log2()

        self.discount: Tensor
        self.register_buffer('discount', discount, persistent=False)

        if not weighted:
            self.register_buffer('idcg', cumsum(1.0 / discount),
                                 persistent=False)
        else:
            self.idcg = None

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        discount = self.discount[:pred_rel_mat.size(1)].view(1, -1)
        dcg = (pred_rel_mat / discount).sum(dim=-1)

        if not self.weighted:
            assert self.idcg is not None
            idcg = self.idcg[data.label_count.clamp(max=self.k)]
        else:
            assert data.edge_label_weight is not None
            pos = data.edge_label_weight_pos
            assert pos is not None

            discount = torch.cat([
                self.discount,
                self.discount.new_full((1, ), fill_value=float('inf')),
            ])
            discount = discount[pos.clamp(max=self.k)]

            idcg = scatter(  # Apply discount and aggregate:
                data.edge_label_weight / discount,
                data.edge_label_index[0],
                dim_size=data.pred_index_mat.size(0),
                reduce='sum',
            )

        out = dcg / idcg
        out[out.isnan() | out.isinf()] = 0.0
        return out


class LinkPredMRR(LinkPredMetric):
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        device = pred_rel_mat.device
        arange = torch.arange(1, pred_rel_mat.size(1) + 1, device=device)
        return (pred_rel_mat / arange).max(dim=-1)[0]


class LinkPredHitRatio(LinkPredMetric):
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        return pred_rel_mat.max(dim=-1)[0].to(torch.get_default_dtype())


class LinkPredCoverage(_LinkPredMetric):
    higher_is_better: bool = True

    def __init__(self, k: int, num_dst_nodes: int) -> None:
        super().__init__(k)
        self.num_dst_nodes = num_dst_nodes

        self.mask: Tensor
        mask = torch.zeros(num_dst_nodes, dtype=torch.bool)
        if WITH_TORCHMETRICS:
            self.add_state('mask', mask, dist_reduce_fx='max')
        else:
            self.register_buffer('mask', mask, persistent=False)

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        self.mask[pred_index_mat[:, :self.k].flatten()] = True

    def compute(self) -> Tensor:
        pass

    def _reset(self) -> None:
        self.mask.zero_()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k={self.k}, '
                f'num_dst_nodes={self.num_dst_nodes})')


class LinkPredDiversity(_LinkPredMetric):
    higher_is_better: bool = True

    def __init__(self, k: int, category: Tensor) -> None:
        super().__init__(k)

        self.accum: Tensor
        self.total: Tensor

        if WITH_TORCHMETRICS:
            self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.register_buffer('accum', torch.tensor(0.), persistent=False)
            self.register_buffer('total', torch.tensor(0), persistent=False)

        self.category: Tensor
        self.register_buffer('category', category, persistent=False)

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        category = self.category[pred_index_mat[:, :self.k]]

        sim = (category.unsqueeze(-2) == category.unsqueeze(-1)).sum(dim=-1)
        div = 1 - 1 / (self.k * (self.k - 1)) * (sim - 1).sum(dim=-1)

        self.accum += div.sum()
        self.total += pred_index_mat.size(0)

    def compute(self) -> Tensor:
        pass

    def _reset(self) -> None:
        self.accum.zero_()
        self.total.zero_()


class LinkPredPersonalization(_LinkPredMetric):
    higher_is_better: bool = True

    def __init__(
        self,
        k: int,
        max_src_nodes: Optional[int] = 2**12,
        batch_size: int = 2**16,
    ) -> None:
        super().__init__(k)
        self.max_src_nodes = max_src_nodes
        self.batch_size = batch_size

        self.preds: List[Tensor]
        self.total: Tensor

        if WITH_TORCHMETRICS:
            self.add_state('preds', default=[], dist_reduce_fx='cat')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.preds = []
            self.register_buffer('total', torch.tensor(0), persistent=False)

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:

        pred_index_mat = pred_index_mat[:, :self.k].cpu()

        if self.max_src_nodes is None:
            self.preds.append(pred_index_mat)
            self.total += pred_index_mat.size(0)
        elif self.total < self.max_src_nodes:
            remaining = int(self.max_src_nodes - self.total)
            pred_index_mat = pred_index_mat[:remaining]
            self.preds.append(pred_index_mat)
            self.total += pred_index_mat.size(0)

    def compute(self) -> Tensor:
        pass

    def _reset(self) -> None:
        self.preds = []
        self.total.zero_()


class LinkPredAveragePopularity(_LinkPredMetric):
    higher_is_better: bool = False

    def __init__(self, k: int, popularity: Tensor) -> None:
        super().__init__(k)

        self.accum: Tensor
        self.total: Tensor

        if WITH_TORCHMETRICS:
            self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.register_buffer('accum', torch.tensor(0.), persistent=False)
            self.register_buffer('total', torch.tensor(0), persistent=False)

        self.popularity: Tensor
        self.register_buffer('popularity', popularity, persistent=False)

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        pred_index_mat = pred_index_mat[:, :self.k]
        popularity = self.popularity[pred_index_mat]
        popularity = popularity.to(self.accum.dtype).mean(dim=-1)
        self.accum += popularity.sum()
        self.total += popularity.numel()

    def compute(self) -> Tensor:
        pass

    def _reset(self) -> None:
        self.accum.zero_()
        self.total.zero_()
