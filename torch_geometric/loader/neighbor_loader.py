from typing import Callable, Dict, List, Optional, Tuple, Union

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.node_loader import NodeLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor


class NeighborLoader(NodeLoader):
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: Optional[bool] = None,
        neighbor_sampler: Optional[NeighborSampler] = None,
        directed: bool = True,  # Deprecated.
        **kwargs,
    ):
        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")

        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                weight_attr=weight_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
                directed=directed,
            )

        super().__init__(
            data=data,
            node_sampler=neighbor_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
