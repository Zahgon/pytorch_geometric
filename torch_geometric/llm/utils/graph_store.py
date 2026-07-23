from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import FeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.sampler import (
    BidirectionalNeighborSampler,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.utils import index_sort

_EdgeTensorType = Union[Tensor, Tuple[Tensor, Tensor]]


class NeighborSamplingRAGGraphStore(LocalGraphStore):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        feature_store: Optional[FeatureStore] = None,
        **kwargs,
    ):
        """Initializes the graph store.
        Optional feature store and neighbor sampling settings.

        Args:
        feature_store (optional): The feature store to use.
            None if not yet registered.
        **kwargs (optional):
            Additional keyword arguments for neighbor sampling.
        """
        self.feature_store = feature_store
        self.sample_kwargs = kwargs
        self._sampler_is_initialized = False
        self._config: Dict[str, Any] = {}

        self.num_neighbors = None
        super().__init__()

    @property
    def config(self) -> Dict[str, Any]:
        pass

    def _set_from_config(self, config: Dict[str, Any], attr_name: str) -> None:
        pass

    @config.setter  # type: ignore
    def config(self, config: Dict[str, Any]) -> None:
        pass

    def _init_sampler(self) -> None:
        """Initializes neighbor sampler with the registered feature store."""
        if self.feature_store is None:
            raise AttributeError("Feature store not registered yet.")
        assert self.num_neighbors is not None, \
            "Please set num_neighbors through config"
        self.sampler = BidirectionalNeighborSampler(
            data=(self.feature_store, self), num_neighbors=self.num_neighbors,
            **self.sample_kwargs)
        self._sampler_is_initialized = True

    def register_feature_store(self, feature_store: FeatureStore) -> None:
        """Registers a feature store with the graph store.

        :param feature_store: The feature store to register.
        """
        self.feature_store = feature_store
        self._sampler_is_initialized = False

    def put_edge_id(  # type: ignore[no-untyped-def]
            self, edge_id: Tensor, *args, **kwargs) -> bool:
        """Stores an edge ID in the graph store.

        :param edge_id: The edge ID to store.
        :return: Whether the operation was successful.
        """
        ret = super().put_edge_id(edge_id.contiguous(), *args, **kwargs)
        self._sampler_is_initialized = False
        return ret

    @property
    def edge_index(self) -> _EdgeTensorType:
        pass

    def put_edge_index(  # type: ignore[no-untyped-def]
            self, edge_index: _EdgeTensorType, *args, **kwargs) -> bool:
        """Stores an edge index in the graph store.

        :param edge_index: The edge index to store.
        :return: Whether the operation was successful.
        """
        ret = super().put_edge_index(edge_index, *args, **kwargs)
        self.edge_idx_args = args
        self.edge_idx_kwargs = kwargs
        self._sampler_is_initialized = False
        return ret

    @edge_index.setter  # type: ignore
    def edge_index(self, edge_index: _EdgeTensorType) -> None:
        pass

    def sample_subgraph(
        self,
        seed_nodes: Tensor,
    ) -> SamplerOutput:
        """Sample the graph starting from the given nodes using the
        in-built NeighborSampler.

        Args:
            seed_nodes (InputNodes): Seed nodes to start sampling from.
            num_neighbors (Optional[NumNeighborsType], optional): Parameters
                to determine how many hops and number of neighbors per hop.
                Defaults to None.

        Returns:
            Union[SamplerOutput, HeteroSamplerOutput]: NeighborSamplerOutput
                for the input.
        """
        if not self._sampler_is_initialized:
            self._init_sampler()

        seed_nodes = seed_nodes.unique().contiguous()
        node_sample_input = NodeSamplerInput(input_id=None, node=seed_nodes)
        out = self.sampler.sample_from_nodes(  # type: ignore[has-type]
            node_sample_input)

        out.edge = self.perm[out.edge]

        return out
