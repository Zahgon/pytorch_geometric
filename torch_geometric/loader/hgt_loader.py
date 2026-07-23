from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor

from torch_geometric.data import FeatureStore, GraphStore, HeteroData
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler import HGTSampler
from torch_geometric.typing import NodeType


class HGTLoader(NodeLoader):
    def __init__(
        self,
        data: Union[HeteroData, Tuple[FeatureStore, GraphStore]],
        num_samples: Union[List[int], Dict[NodeType, List[int]]],
        input_nodes: Union[NodeType, Tuple[NodeType, Optional[Tensor]]],
        is_sorted: bool = False,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        **kwargs,
    ):
        hgt_sampler = HGTSampler(
            data,
            num_samples=num_samples,
            is_sorted=is_sorted,
            share_memory=kwargs.get('num_workers', 0) > 0,
        )

        super().__init__(
            data=data,
            node_sampler=hgt_sampler,
            input_nodes=input_nodes,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
