from typing import Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import torch

from torch_geometric.distributed import (
    DistContext,
    DistLoader,
    DistNeighborSampler,
    LocalFeatureStore,
    LocalGraphStore,
)
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor


class DistNeighborLoader(NodeLoader, DistLoader):
    def __init__(
        self,
        data: Tuple[LocalFeatureStore, LocalGraphStore],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        master_addr: str,
        master_port: Union[int, str],
        current_ctx: DistContext,
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        dist_sampler: Optional[DistNeighborSampler] = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        concurrency: int = 1,
        num_rpc_threads: int = 16,
        filter_per_worker: Optional[bool] = False,
        async_sampling: bool = True,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        assert isinstance(data[0], LocalFeatureStore)
        assert isinstance(data[1], LocalGraphStore)
        assert concurrency >= 1, "RPC concurrency must be greater than 1"

        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")

        channel = torch.multiprocessing.Queue() if async_sampling else None

        if dist_sampler is None:
            dist_sampler = DistNeighborSampler(
                data=data,
                current_ctx=current_ctx,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                device=device,
                channel=channel,
                concurrency=concurrency,
            )
        else:
            warn(  # noqa: B028
                "`torch_geometric.distributed` has been deprecated since 2.7.0 and will "  # noqa: E501
                "no longer be maintained. For distributed training, refer to our "  # noqa: E501
                "tutorials on distributed training at "
                "https://pytorch-geometric.readthedocs.io/en/latest/tutorial/distributed.html "  # noqa: E501
                "or cuGraph examples at "
                "https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg/cugraph_pyg/examples",  # noqa: E501
                stack_level=2)

        DistLoader.__init__(
            self,
            channel=channel,
            master_addr=master_addr,
            master_port=master_port,
            current_ctx=current_ctx,
            dist_sampler=dist_sampler,
            num_rpc_threads=num_rpc_threads,
            **kwargs,
        )
        NodeLoader.__init__(
            self,
            data=data,
            node_sampler=dist_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            filter_per_worker=filter_per_worker,
            transform_sampler_output=self.channel_get if channel else None,
            worker_init_fn=self.worker_init_fn,
            **kwargs,
        )

    def __repr__(self) -> str:
        return DistLoader.__repr__(self)
