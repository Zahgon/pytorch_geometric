from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from torch_geometric.utils import coalesce, to_undirected

LOSS_TYPE = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]


class PRBCDAttack(torch.nn.Module):
    coeffs = {
        'max_final_samples': 20,
        'max_trials_sampling': 20,
        'with_early_stopping': True,
        'eps': 1e-7
    }

    def __init__(
        self,
        model: torch.nn.Module,
        block_size: int,
        epochs: int = 125,
        epochs_resampling: int = 100,
        loss: Optional[Union[str, LOSS_TYPE]] = 'prob_margin',
        metric: Optional[Union[str, LOSS_TYPE]] = None,
        lr: float = 1_000,
        is_undirected: bool = True,
        log: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.block_size = block_size
        self.epochs = epochs

        if isinstance(loss, str):
            if loss == 'masked':
                self.loss = self._masked_cross_entropy
            elif loss == 'margin':
                self.loss = partial(self._margin_loss, reduce='mean')
            elif loss == 'prob_margin':
                self.loss = self._probability_margin_loss
            elif loss == 'tanh_margin':
                self.loss = self._tanh_margin_loss
            else:
                raise ValueError(f'Unknown loss `{loss}`')
        else:
            self.loss = loss

        self.is_undirected = is_undirected
        self.log = log
        self.metric = metric or self.loss

        self.epochs_resampling = epochs_resampling
        self.lr = lr

        self.coeffs.update(kwargs)

    def attack(
        self,
        x: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        pass

    def _prepare(self, budget: int) -> Iterable[int]:
        pass

    @torch.no_grad()
    def _update(self, epoch: int, gradient: Tensor, x: Tensor, labels: Tensor,
                budget: int, idx_attack: Optional[Tensor] = None,
                **kwargs) -> Dict[str, float]:
        """Update edge weights given gradient."""
        self.block_edge_weight = self._update_edge_weights(
            budget, self.block_edge_weight, epoch, gradient)

        pmass_update = torch.clamp(self.block_edge_weight, 0, 1)
        self.block_edge_weight = self._project(budget, self.block_edge_weight,
                                               self.coeffs['eps'])

        scalars = dict(
            prob_mass_after_update=pmass_update.sum().item(),
            prob_mass_after_update_max=pmass_update.max().item(),
            prob_mass_after_projection=self.block_edge_weight.sum().item(),
            prob_mass_after_projection_nonzero_weights=(
                self.block_edge_weight > self.coeffs['eps']).sum().item(),
            prob_mass_after_projection_max=self.block_edge_weight.max().item())
        if not self.coeffs['with_early_stopping']:
            return scalars

        topk_block_edge_weight = torch.zeros_like(self.block_edge_weight)
        topk_block_edge_weight[torch.topk(self.block_edge_weight,
                                          budget).indices] = 1
        edge_index, edge_weight = self._get_modified_adj(
            self.edge_index, self.edge_weight, self.block_edge_index,
            topk_block_edge_weight)
        prediction = self._forward(x, edge_index, edge_weight, **kwargs)
        metric = self.metric(prediction, labels, idx_attack)

        if metric > self.best_metric:
            self.best_metric = metric
            self.best_block = self.current_block.cpu().clone()
            self.best_edge_index = self.block_edge_index.cpu().clone()
            self.best_pert_edge_weight = self.block_edge_weight.cpu().clone()

        if epoch < self.epochs_resampling - 1:
            self._resample_random_block(budget)
        elif epoch == self.epochs_resampling - 1:
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            block_edge_weight = self.best_pert_edge_weight.clone()
            self.block_edge_weight = block_edge_weight.to(self.device)

        scalars['metric'] = metric.item()
        return scalars

    @torch.no_grad()
    def _close(self, x: Tensor, labels: Tensor, budget: int,
               idx_attack: Optional[Tensor] = None,
               **kwargs) -> Tuple[Tensor, Tensor]:
        pass

    def _forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                 **kwargs) -> Tensor:
        """Forward model."""
        return self.model(x, edge_index, edge_weight, **kwargs)

    def _forward_and_gradient(self, x: Tensor, labels: Tensor,
                              idx_attack: Optional[Tensor] = None,
                              **kwargs) -> Tuple[Tensor, Tensor]:
        pass

    def _get_modified_adj(self, edge_index: Tensor, edge_weight: Tensor,
                          block_edge_index: Tensor,
                          block_edge_weight: Tensor) -> Tuple[Tensor, Tensor]:
        """Merges adjacency matrix with current block (incl. weights)."""
        if self.is_undirected:
            block_edge_index, block_edge_weight = to_undirected(
                block_edge_index, block_edge_weight, num_nodes=self.num_nodes,
                reduce='mean')

        modified_edge_index = torch.cat(
            (edge_index.to(self.device), block_edge_index), dim=-1)
        modified_edge_weight = torch.cat(
            (edge_weight.to(self.device), block_edge_weight))

        modified_edge_index, modified_edge_weight = coalesce(
            modified_edge_index, modified_edge_weight,
            num_nodes=self.num_nodes, reduce='sum')

        is_edge_in_clean_adj = modified_edge_weight > 1
        modified_edge_weight[is_edge_in_clean_adj] = (
            2 - modified_edge_weight[is_edge_in_clean_adj])

        return modified_edge_index, modified_edge_weight

    def _filter_self_loops_in_block(self, with_weight: bool):
        is_not_sl = self.block_edge_index[0] != self.block_edge_index[1]
        self.current_block = self.current_block[is_not_sl]
        self.block_edge_index = self.block_edge_index[:, is_not_sl]
        if with_weight:
            self.block_edge_weight = self.block_edge_weight[is_not_sl]

    def _sample_random_block(self, budget: int = 0):
        for _ in range(self.coeffs['max_trials_sampling']):
            num_possible_edges = self._num_possible_edges(
                self.num_nodes, self.is_undirected)
            self.current_block = torch.randint(num_possible_edges,
                                               (self.block_size, ),
                                               device=self.device)
            self.current_block = torch.unique(self.current_block, sorted=True)
            if self.is_undirected:
                self.block_edge_index = self._linear_to_triu_idx(
                    self.num_nodes, self.current_block)
            else:
                self.block_edge_index = self._linear_to_full_idx(
                    self.num_nodes, self.current_block)
                self._filter_self_loops_in_block(with_weight=False)

            self.block_edge_weight = torch.full(self.current_block.shape,
                                                self.coeffs['eps'],
                                                device=self.device)
            if self.current_block.size(0) >= budget:
                return
        raise RuntimeError('Sampling random block was not successful. '
                           'Please decrease `budget`.')

    def _resample_random_block(self, budget: int):
        sorted_idx = torch.argsort(self.block_edge_weight)
        keep_above = (self.block_edge_weight
                      <= self.coeffs['eps']).sum().long()
        if keep_above < sorted_idx.size(0) // 2:
            keep_above = sorted_idx.size(0) // 2
        sorted_idx = sorted_idx[keep_above:]

        self.current_block = self.current_block[sorted_idx]

        for _ in range(self.coeffs['max_trials_sampling']):
            n_edges_resample = self.block_size - self.current_block.size(0)
            num_possible_edges = self._num_possible_edges(
                self.num_nodes, self.is_undirected)
            lin_index = torch.randint(num_possible_edges, (n_edges_resample, ),
                                      device=self.device)

            current_block = torch.cat((self.current_block, lin_index))
            self.current_block, unique_idx = torch.unique(
                current_block, sorted=True, return_inverse=True)

            if self.is_undirected:
                self.block_edge_index = self._linear_to_triu_idx(
                    self.num_nodes, self.current_block)
            else:
                self.block_edge_index = self._linear_to_full_idx(
                    self.num_nodes, self.current_block)

            block_edge_weight_prev = self.block_edge_weight[sorted_idx]
            self.block_edge_weight = torch.full(self.current_block.shape,
                                                self.coeffs['eps'],
                                                device=self.device)
            self.block_edge_weight[
                unique_idx[:sorted_idx.size(0)]] = block_edge_weight_prev

            if not self.is_undirected:
                self._filter_self_loops_in_block(with_weight=True)

            if self.current_block.size(0) > budget:
                return
        raise RuntimeError('Sampling random block was not successful.'
                           'Please decrease `budget`.')

    def _sample_final_edges(self, x: Tensor, labels: Tensor, budget: int,
                            idx_attack: Optional[Tensor] = None,
                            **kwargs) -> Tuple[Tensor, Tensor]:
        pass

    def _update_edge_weights(self, budget: int, block_edge_weight: Tensor,
                             epoch: int, gradient: Tensor) -> Tensor:
        lr = (budget / self.num_nodes * self.lr /
              np.sqrt(max(0, epoch - self.epochs_resampling) + 1))
        return block_edge_weight + lr * gradient

    @staticmethod
    def _project(budget: int, values: Tensor, eps: float = 1e-7) -> Tensor:
        r"""Project :obj:`values`:
        :math:`budget \ge \sum \Pi_{[0, 1]}(\text{values})`.
        """
        if torch.clamp(values, 0, 1).sum() > budget:
            left = (values - 1).min()
            right = values.max()
            miu = PRBCDAttack._bisection(values, left, right, budget)
            values = values - miu
        return torch.clamp(values, min=eps, max=1 - eps)

    @staticmethod
    def _bisection(edge_weights: Tensor, a: float, b: float, n_pert: int,
                   eps=1e-5, max_iter=1e3) -> Tensor:
        """Bisection search for projection."""
        def shift(offset: float):
            return (torch.clamp(edge_weights - offset, 0, 1).sum() - n_pert)

        miu = a
        for _ in range(int(max_iter)):
            miu = (a + b) / 2
            if (shift(miu) == 0.0):
                break
            if (shift(miu) * shift(a) < 0):
                b = miu
            else:
                a = miu
            if ((b - a) <= eps):
                break
        return miu

    @staticmethod
    def _num_possible_edges(n: int, is_undirected: bool) -> int:
        """Determine number of possible edges for graph."""
        if is_undirected:
            return n * (n - 1) // 2
        else:
            return int(n**2)  # We filter self-loops later

    @staticmethod
    def _linear_to_triu_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Linear index to upper triangular matrix without diagonal. This is
        similar to
        https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients/28116498#28116498
        with number nodes decremented and col index incremented by one.
        """
        nn = n * (n - 1)
        row_idx = n - 2 - torch.floor(
            torch.sqrt(-8 * lin_idx.double() + 4 * nn - 7) / 2.0 - 0.5).long()
        col_idx = 1 + lin_idx + row_idx - nn // 2 + torch.div(
            (n - row_idx) * (n - row_idx - 1), 2, rounding_mode='floor')
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def _linear_to_full_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Linear index to dense matrix including diagonal."""
        row_idx = torch.div(lin_idx, n, rounding_mode='floor')
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def _margin_loss(score: Tensor, labels: Tensor,
                     idx_mask: Optional[Tensor] = None,
                     reduce: Optional[str] = None) -> Tensor:
        pass

    @staticmethod
    def _tanh_margin_loss(prediction: Tensor, labels: Tensor,
                          idx_mask: Optional[Tensor] = None) -> Tensor:
        pass

    @staticmethod
    def _probability_margin_loss(prediction: Tensor, labels: Tensor,
                                 idx_mask: Optional[Tensor] = None) -> Tensor:
        pass

    @staticmethod
    def _masked_cross_entropy(log_prob: Tensor, labels: Tensor,
                              idx_mask: Optional[Tensor] = None) -> Tensor:
        pass

    def _append_statistics(self, mapping: Dict[str, Any]):
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class GRBCDAttack(PRBCDAttack):
    coeffs = {'max_trials_sampling': 20, 'eps': 1e-7}

    def __init__(
        self,
        model: torch.nn.Module,
        block_size: int,
        epochs: int = 125,
        loss: Optional[Union[str, LOSS_TYPE]] = 'masked',
        is_undirected: bool = True,
        log: bool = True,
        **kwargs,
    ):
        super().__init__(model, block_size, epochs, loss=loss,
                         is_undirected=is_undirected, log=log, **kwargs)

    @torch.no_grad()
    def _prepare(self, budget: int) -> List[int]:
        pass

    @torch.no_grad()
    def _update(self, step_size: int, gradient: Tensor, *args,
                **kwargs) -> Dict[str, Any]:
        """Update edge weights given gradient."""
        _, topk_edge_index = torch.topk(gradient, step_size)

        flip_edge_index = self.block_edge_index[:, topk_edge_index]
        flip_edge_weight = torch.ones_like(flip_edge_index[0],
                                           dtype=torch.float32)

        self.flipped_edges = torch.cat((self.flipped_edges, flip_edge_index),
                                       axis=-1)

        if self.is_undirected:
            flip_edge_index, flip_edge_weight = to_undirected(
                flip_edge_index, flip_edge_weight, num_nodes=self.num_nodes,
                reduce='mean')
        edge_index = torch.cat(
            (self.edge_index.to(self.device), flip_edge_index.to(self.device)),
            dim=-1)
        edge_weight = torch.cat((self.edge_weight.to(self.device),
                                 flip_edge_weight.to(self.device)))
        edge_index, edge_weight = coalesce(edge_index, edge_weight,
                                           num_nodes=self.num_nodes,
                                           reduce='sum')

        is_one_mask = torch.isclose(edge_weight, torch.tensor(1.))
        self.edge_index = edge_index[:, is_one_mask]
        self.edge_weight = edge_weight[is_one_mask]
        assert self.edge_index.size(1) == self.edge_weight.size(0)

        self._sample_random_block(step_size)

        scalars = {
            'number_positive_entries_in_gradient': (gradient > 0).sum().item()
        }
        return scalars

    def _close(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        pass
