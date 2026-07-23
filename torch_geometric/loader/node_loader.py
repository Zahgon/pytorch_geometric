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
    get_input_nodes,
    infer_filter_per_worker,
)
from torch_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.typing import InputNodes, OptTensor


class NodeLoader(
        torch.utils.data.DataLoader,
        AffinityMixin,
        MultithreadingMixin,
        LogMemoryMixin,
):
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        node_sampler: BaseSampler,
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional[HeteroData] = None,
        input_id: OptTensor = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        self.data = data
        self.node_sampler = node_sampler
        self.input_nodes = input_nodes
        self.input_time = input_time
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls
        self.input_id = input_id

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        input_type, input_nodes, input_id = get_input_nodes(
            data, input_nodes, input_id)

        self.input_data = NodeSamplerInput(
            input_id=input_id,
            node=input_nodes,
            time=input_time,
            input_type=input_type,
        )

        iterator = range(input_nodes.size(0))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input nodes."""
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
