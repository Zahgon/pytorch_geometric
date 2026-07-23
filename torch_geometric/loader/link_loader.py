from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.mixin import (
    AffinityMixin,
    LogMemoryMixin,
    MultithreadingMixin,
)
from torch_geometric.loader.utils import (
    filter_custom_hetero_store,
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    get_edge_label_index,
    infer_filter_per_worker,
)
from torch_geometric.sampler import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NegativeSampling,
    SamplerOutput,
)
from torch_geometric.typing import InputEdges, OptTensor


class LinkLoader(
        torch.utils.data.DataLoader,
        AffinityMixin,
        MultithreadingMixin,
        LogMemoryMixin,
):
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        link_sampler: BaseSampler,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        edge_label_time: OptTensor = None,
        neg_sampling: Optional[NegativeSampling] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional[HeteroData] = None,
        input_id: OptTensor = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)
        self.edge_label_index = edge_label_index

        if neg_sampling_ratio is not None and neg_sampling_ratio != 0.0:
            neg_sampling = NegativeSampling("binary", neg_sampling_ratio)

        input_type, edge_label_index = get_edge_label_index(
            data, edge_label_index)

        self.data = data
        self.link_sampler = link_sampler
        self.neg_sampling = NegativeSampling.cast(neg_sampling)
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls

        if (self.neg_sampling is not None and self.neg_sampling.is_binary()
                and edge_label is not None and edge_label.min() == 0):
            edge_label = edge_label + 1

        if (self.neg_sampling is not None and self.neg_sampling.is_triplet()
                and edge_label is not None):
            raise ValueError("'edge_label' needs to be undefined for "
                             "'triplet'-based negative sampling. Please use "
                             "`src_index`, `dst_pos_index` and "
                             "`neg_pos_index` of the returned mini-batch "
                             "instead to differentiate between positive and "
                             "negative samples.")

        self.input_data = EdgeSamplerInput(
            input_id=input_id,
            row=edge_label_index[0],
            col=edge_label_index[1],
            label=edge_label,
            time=edge_label_time,
            input_type=input_type,
        )

        iterator = range(edge_label_index.size(1))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input edges."""
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def collate_fn(self, index: Union[Tensor, List[int]]) -> Any:
        pass

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        pass

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()

        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
