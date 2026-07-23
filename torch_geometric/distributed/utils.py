from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.sampler import SamplerOutput
from torch_geometric.typing import EdgeType, NodeType


@dataclass
class DistEdgeHeteroSamplerInput:
    input_id: Optional[Tensor]
    node_dict: Dict[NodeType, Tensor]
    time_dict: Optional[Dict[NodeType, Tensor]] = None
    input_type: Optional[EdgeType] = None


class NodeDict:
    def __init__(self, node_types, num_hops):
        self.src: Dict[NodeType, List[Tensor]] = {
            k: (num_hops + 1) * [torch.empty(0, dtype=torch.int64)]
            for k in node_types
        }
        self.with_dupl: Dict[NodeType, Tensor] = {
            k: torch.empty(0, dtype=torch.int64)
            for k in node_types
        }
        self.out: Dict[NodeType, Tensor] = {
            k: torch.empty(0, dtype=torch.int64)
            for k in node_types
        }
        self.seed_time: Dict[NodeType, List[Tensor]] = {
            k: num_hops * [torch.empty(0, dtype=torch.int64)]
            for k in node_types
        }


class BatchDict:
    def __init__(self, node_types, num_hops):
        self.src: Dict[NodeType, List[Tensor]] = {
            k: (num_hops + 1) * [torch.empty(0, dtype=torch.int64)]
            for k in node_types
        }
        self.with_dupl: Dict[NodeType, Tensor] = {
            k: torch.empty(0, dtype=torch.int64)
            for k in node_types
        }
        self.out: Dict[NodeType, Tensor] = {
            k: torch.empty(0, dtype=torch.int64)
            for k in node_types
        }


def remove_duplicates(
    out: SamplerOutput,
    node: Tensor,
    batch: Optional[Tensor] = None,
    disjoint: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    num_nodes = node.numel()
    node_combined = torch.cat([node, out.node])

    if not disjoint:
        _, idx = np.unique(node_combined.cpu().numpy(), return_index=True)
        idx = torch.from_numpy(idx).to(node.device).sort().values

        node = node_combined[idx]
        src = node[num_nodes:]

        return (src, node, None, None)

    else:
        batch_combined = torch.cat([batch, out.batch])
        node_batch = torch.stack([batch_combined, node_combined], dim=0)

        _, idx = np.unique(node_batch.cpu().numpy(), axis=1, return_index=True)
        idx = torch.from_numpy(idx).to(node.device).sort().values

        batch = batch_combined[idx]
        node = node_combined[idx]
        src_batch = batch[num_nodes:]
        src = node[num_nodes:]

        return (src, node, src_batch, batch)


def filter_dist_store(
    feature_store: LocalFeatureStore,
    graph_store: LocalGraphStore,
    node_dict: Dict[str, Tensor],
    row_dict: Dict[str, Tensor],
    col_dict: Dict[str, Tensor],
    edge_dict: Dict[str, Optional[Tensor]],
    custom_cls: Optional[HeteroData] = None,
    meta: Optional[Dict[str, Tensor]] = None,
    input_type: str = None,
) -> HeteroData:
    pass


def as_str(inputs: Union[NodeType, EdgeType]) -> str:
    if isinstance(inputs, NodeType):
        return inputs
    elif isinstance(inputs, (list, tuple)) and len(inputs) == 3:
        return '__'.join(inputs)
    return ''


def reverse_edge_type(etype: EdgeType) -> EdgeType:
    src, rel, dst = etype
    if src != dst:
        if rel.split('_', 1)[0] == 'rev':
            rel = rel.split('_', 1)[1]
        else:
            rel = 'rev_' + rel

    return dst, rel, src
