import copy
import math
import sys
import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import (
    Data,
    FeatureStore,
    GraphStore,
    HeteroData,
    remote_backend_utils,
)
from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.sampler import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NegativeSampling,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.base import DataType, NumNeighbors, SubgraphType
from torch_geometric.sampler.utils import (
    global_to_local_node_idx,
    remap_keys,
    reverse_edge_type,
    to_csc,
    to_hetero_csc,
)
from torch_geometric.typing import EdgeType, NodeType, OptTensor

NumNeighborsType = Union[NumNeighbors, List[int], Dict[EdgeType, List[int]]]


class NeighborSampler(BaseSampler):
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: NumNeighborsType,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        replace: bool = False,
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        is_sorted: bool = False,
        share_memory: bool = False,
        directed: bool = True,  # Deprecated
        sample_direction: Literal['forward', 'backward'] = 'forward',
    ):
        if not directed:
            subgraph_type = SubgraphType.induced
            warnings.warn(
                f"The usage of the 'directed' argument in "
                f"'{self.__class__.__name__}' is deprecated. Use "
                f"`subgraph_type='induced'` instead.", stacklevel=2)

        if (not torch_geometric.typing.WITH_PYG_LIB and sys.platform == 'linux'
                and subgraph_type != SubgraphType.induced):
            warnings.warn(
                f"Using '{self.__class__.__name__}' without a "
                f"'pyg-lib' installation is deprecated and will be "
                f"removed soon. Please install 'pyg-lib' for "
                f"accelerated neighborhood sampling", stacklevel=2)

        self.data_type = DataType.from_data(data)
        self.sample_direction = sample_direction

        if self.sample_direction == 'backward':
            if time_attr is not None:
                raise NotImplementedError(
                    "Temporal Sampling not yet supported for backward sampling"
                )

        if self.data_type == DataType.homogeneous:
            self.num_nodes = data.num_nodes

            self.node_time: Optional[Tensor] = None
            self.edge_time: Optional[Tensor] = None

            if time_attr is not None:
                if data.is_node_attr(time_attr):
                    self.node_time = data[time_attr]
                elif data.is_edge_attr(time_attr):
                    self.edge_time = data[time_attr]
                else:
                    raise ValueError(
                        f"The time attribute '{time_attr}' is neither a "
                        f"node-level or edge-level attribute")

            self.colptr, self.row, self.perm = to_csc(
                data, device='cpu', share_memory=share_memory,
                is_sorted=is_sorted, src_node_time=self.node_time,
                edge_time=self.edge_time,
                to_transpose=self.sample_direction == 'backward')

            if self.edge_time is not None and self.perm is not None:
                self.edge_time = self.edge_time[self.perm]

            self.edge_weight: Optional[Tensor] = None
            if weight_attr is not None:
                self.edge_weight = data[weight_attr]
                if self.perm is not None:
                    self.edge_weight = self.edge_weight[self.perm]

        elif self.data_type == DataType.heterogeneous:
            self.node_types, self.edge_types = data.metadata()

            if self.sample_direction == 'backward':
                self.edge_types = [
                    reverse_edge_type(edge_type)
                    for edge_type in self.edge_types
                ]
                self.to_restored_edge_type = {
                    k: reverse_edge_type(k)
                    for k in self.edge_types
                }

            self.num_nodes = {k: data[k].num_nodes for k in self.node_types}

            self.node_time: Optional[Dict[NodeType, Tensor]] = None
            self.edge_time: Optional[Dict[EdgeType, Tensor]] = None

            if time_attr is not None:
                is_node_level_time = is_edge_level_time = False

                for store in data.node_stores:
                    if time_attr in store:
                        is_node_level_time = True
                for store in data.edge_stores:
                    if time_attr in store:
                        is_edge_level_time = True

                if is_node_level_time and is_edge_level_time:
                    raise ValueError(
                        f"The time attribute '{time_attr}' holds both "
                        f"node-level and edge-level information")

                if not is_node_level_time and not is_edge_level_time:
                    raise ValueError(
                        f"The time attribute '{time_attr}' is neither a "
                        f"node-level or edge-level attribute")

                if is_node_level_time:
                    self.node_time = data.collect(time_attr)
                else:
                    self.edge_time = data.collect(time_attr)

            self.to_rel_type = {k: '__'.join(k) for k in self.edge_types}
            self.to_edge_type = {v: k for k, v in self.to_rel_type.items()}

            colptr_dict, row_dict, self.perm = to_hetero_csc(
                data, device='cpu', share_memory=share_memory,
                is_sorted=is_sorted, node_time_dict=self.node_time,
                edge_time_dict=self.edge_time,
                to_transpose=self.sample_direction == 'backward')

            self.row_dict = remap_keys(row_dict, self.to_rel_type)
            self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)

            if self.edge_time is not None:
                for edge_type, edge_time in self.edge_time.items():
                    if self.perm.get(edge_type, None) is not None:
                        edge_time = edge_time[self.perm[edge_type]]
                        self.edge_time[edge_type] = edge_time
                self.edge_time = remap_keys(self.edge_time, self.to_rel_type)

            self.edge_weight: Optional[Dict[EdgeType, Tensor]] = None
            if weight_attr is not None:
                self.edge_weight = data.collect(weight_attr)
                for edge_type, edge_weight in self.edge_weight.items():
                    if self.perm.get(edge_type, None) is not None:
                        edge_weight = edge_weight[self.perm[edge_type]]
                        self.edge_weight[edge_type] = edge_weight
                self.edge_weight = remap_keys(self.edge_weight,
                                              self.to_rel_type)

        else:  # self.data_type == DataType.remote
            feature_store, graph_store = data

            attrs = [attr for attr in feature_store.get_all_tensor_attrs()]

            edge_attrs = graph_store.get_all_edge_attrs()
            self.edge_types = list({attr.edge_type for attr in edge_attrs})

            if self.sample_direction == 'backward':
                self.edge_types = [
                    reverse_edge_type(edge_type)
                    for edge_type in self.edge_types
                ]
                self.to_restored_edge_type = {
                    k: reverse_edge_type(k)
                    for k in self.edge_types
                }
                self.to_backward_edge_type = {
                    v: k
                    for k, v in self.to_restored_edge_type.items()
                }

            if weight_attr is not None:
                raise NotImplementedError(
                    f"'weight_attr' argument not yet supported within "
                    f"'{self.__class__.__name__}' for "
                    f"'(FeatureStore, GraphStore)' inputs")

            if time_attr is not None:
                for edge_attr in edge_attrs:
                    if edge_attr.layout == EdgeLayout.CSR:
                        raise ValueError(
                            "Temporal sampling requires that edges are stored "
                            "in either COO or CSC layout")
                    if not edge_attr.is_sorted:
                        raise ValueError(
                            "Temporal sampling requires that edges are "
                            "sorted by destination, and by source time "
                            "within local neighborhoods")

                time_attrs = [
                    copy.copy(attr) for attr in attrs
                    if attr.attr_name == time_attr
                ]

            if not self.is_hetero:
                self.node_types = [None]
                self.num_nodes = max(edge_attrs[0].size)
                self.edge_weight: Optional[Tensor] = None

                self.node_time: Optional[Tensor] = None
                self.edge_time: Optional[Tensor] = None

                if time_attr is not None:
                    if len(time_attrs) != 1:
                        raise ValueError("Temporal sampling specified but did "
                                         "not find any temporal data")
                    time_attrs[0].index = None  # Reset index for full data.
                    time_tensor = feature_store.get_tensor(time_attrs[0])
                    if time_attr == 'time':
                        self.node_time = time_tensor
                    else:
                        self.edge_time = time_tensor

                if self.sample_direction == 'forward':
                    self.row, self.colptr, self.perm = graph_store.csc()
                elif self.sample_direction == 'backward':
                    self.colptr, self.row, self.perm = graph_store.csr()

            else:
                node_types = [
                    attr.group_name for attr in attrs
                    if isinstance(attr.group_name, str)
                ]
                self.node_types = list(set(node_types))
                self.num_nodes = {
                    node_type: remote_backend_utils.size(*data, node_type)
                    for node_type in self.node_types
                }
                self.edge_weight: Optional[Dict[EdgeType, Tensor]] = None

                self.node_time: Optional[Dict[NodeType, Tensor]] = None
                self.edge_time: Optional[Dict[EdgeType, Tensor]] = None

                if time_attr is not None:
                    for attr in time_attrs:  # Reset index for full data.
                        attr.index = None

                    time_tensors = feature_store.multi_get_tensor(time_attrs)
                    time = {
                        attr.group_name: time_tensor
                        for attr, time_tensor in zip(time_attrs, time_tensors)
                    }

                    group_names = [attr.group_name for attr in time_attrs]
                    if all([isinstance(g, str) for g in group_names]):
                        self.node_time = time
                    elif all([isinstance(g, tuple) for g in group_names]):
                        self.edge_time = time
                    else:
                        raise ValueError(
                            f"Found time attribute '{time_attr}' for both "
                            f"node-level and edge-level types")

                self.to_rel_type = {k: '__'.join(k) for k in self.edge_types}
                self.to_edge_type = {v: k for k, v in self.to_rel_type.items()}
                if self.sample_direction == 'forward':
                    row_dict, colptr_dict, self.perm = graph_store.csc()
                elif self.sample_direction == 'backward':
                    colptr_dict, row_dict, self.perm = graph_store.csr()

                    colptr_dict = remap_keys(colptr_dict,
                                             self.to_backward_edge_type)
                    row_dict = remap_keys(row_dict, self.to_backward_edge_type)
                    self.perm = remap_keys(self.perm,
                                           self.to_backward_edge_type)

                self.row_dict = remap_keys(row_dict, self.to_rel_type)
                self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)

        if (self.edge_time is not None
                and not torch_geometric.typing.WITH_EDGE_TIME_NEIGHBOR_SAMPLE):
            raise ImportError("Edge-level temporal sampling requires a "
                              "more recent 'pyg-lib' installation")

        if (self.edge_weight is not None
                and not torch_geometric.typing.WITH_WEIGHTED_NEIGHBOR_SAMPLE):
            raise ImportError("Weighted neighbor sampling requires "
                              "'pyg-lib>=0.3.0'")

        self.num_neighbors = num_neighbors
        self.replace = replace
        self.subgraph_type = SubgraphType(subgraph_type)
        self.disjoint = disjoint
        self.temporal_strategy = temporal_strategy
        self.keep_orig_edges = False

    @property
    def num_neighbors(self) -> NumNeighbors:
        pass

    @num_neighbors.setter
    def num_neighbors(self, num_neighbors: NumNeighborsType):
        pass

    @property
    def is_hetero(self) -> bool:
        pass

    @property
    def is_temporal(self) -> bool:
        pass

    @property
    def disjoint(self) -> bool:
        pass

    @disjoint.setter
    def disjoint(self, disjoint: bool):
        pass


    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        out = node_sample(inputs, self._sample)
        if self.subgraph_type == SubgraphType.bidirectional:
            out = out.to_bidirectional(keep_orig_edges=self.keep_orig_edges)
        return out


    def sample_from_edges(
        self,
        inputs: EdgeSamplerInput,
        neg_sampling: Optional[NegativeSampling] = None,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        pass


    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        pass


    def _sample(
        self,
        seed: Union[Tensor, Dict[NodeType, Tensor]],
        seed_time: Optional[Union[Tensor, Dict[NodeType, Tensor]]] = None,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Implements neighbor sampling by calling either :obj:`pyg-lib` (if
        installed) or :obj:`torch-sparse` (if installed) sampling routines.
        """
        if isinstance(seed, dict):  # Heterogeneous sampling:
            if (torch_geometric.typing.WITH_PYG_LIB
                    and self.subgraph_type != SubgraphType.induced):
                colptrs = list(self.colptr_dict.values())
                dtype = colptrs[0].dtype if len(colptrs) > 0 else torch.int64
                seed = {k: v.to(dtype) for k, v in seed.items()}

                args = (
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    seed,
                    self.num_neighbors.get_mapped_values(self.edge_types),
                    self.node_time,
                )
                if torch_geometric.typing.WITH_EDGE_TIME_NEIGHBOR_SAMPLE:
                    args += (self.edge_time, )
                args += (seed_time, )
                if torch_geometric.typing.WITH_WEIGHTED_NEIGHBOR_SAMPLE:
                    args += (self.edge_weight, )
                args += (
                    True,  # csc
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                    self.disjoint,
                    self.temporal_strategy,
                    True,  # return_edge_id
                )

                out = torch.ops.pyg.hetero_neighbor_sample(*args)
                row, col, node, edge, batch = out[:4] + (None, )

                num_sampled_nodes = num_sampled_edges = None
                if len(out) >= 6:
                    num_sampled_nodes, num_sampled_edges = out[4:6]

                if self.disjoint:
                    node = {k: v.t().contiguous() for k, v in node.items()}
                    batch = {k: v[0] for k, v in node.items()}
                    node = {k: v[1] for k, v in node.items()}

            elif torch_geometric.typing.WITH_TORCH_SPARSE:
                if self.disjoint:
                    if self.subgraph_type == SubgraphType.induced:
                        raise ValueError("'disjoint' sampling not supported "
                                         "for neighbor sampling with "
                                         "`subgraph_type='induced'`")
                    else:
                        raise ValueError("'disjoint' sampling not supported "
                                         "for neighbor sampling via "
                                         "'torch-sparse'. Please install "
                                         "'pyg-lib' for improved and "
                                         "optimized sampling routines.")

                out = torch.ops.torch_sparse.hetero_neighbor_sample(
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    seed,  # seed_dict
                    self.num_neighbors.get_mapped_values(self.edge_types),
                    self.num_neighbors.num_hops,
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                )
                node, row, col, edge, batch = out + (None, )
                num_sampled_nodes = num_sampled_edges = None

            else:
                raise ImportError(f"'{self.__class__.__name__}' requires "
                                  f"either 'pyg-lib' or 'torch-sparse'")

            if self.sample_direction == 'backward':
                row, col = col, row

            row = remap_keys(row, self.to_edge_type)
            col = remap_keys(col, self.to_edge_type)
            edge = remap_keys(edge, self.to_edge_type)

            if self.sample_direction == 'backward':
                row = remap_keys(row, self.to_restored_edge_type)
                col = remap_keys(col, self.to_restored_edge_type)
                edge = remap_keys(edge, self.to_restored_edge_type)

            if num_sampled_edges is not None:
                num_sampled_edges = remap_keys(
                    num_sampled_edges,
                    self.to_edge_type,
                )
                if self.sample_direction == 'backward':
                    num_sampled_edges = remap_keys(num_sampled_edges,
                                                   self.to_restored_edge_type)

            return HeteroSamplerOutput(
                node=node,
                row=row,
                col=col,
                edge=edge,
                batch=batch,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
            )

        else:  # Homogeneous sampling:
            if (torch_geometric.typing.WITH_PYG_LIB
                    and self.subgraph_type != SubgraphType.induced):

                args = (
                    self.colptr,
                    self.row,
                    seed.to(self.colptr.dtype),
                    self.num_neighbors.get_mapped_values(),
                    self.node_time,
                )
                if torch_geometric.typing.WITH_EDGE_TIME_NEIGHBOR_SAMPLE:
                    args += (self.edge_time, )
                args += (seed_time, )
                if torch_geometric.typing.WITH_WEIGHTED_NEIGHBOR_SAMPLE:
                    args += (self.edge_weight, )
                args += (
                    True,  # csc
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                    self.disjoint,
                    self.temporal_strategy,
                    True,  # return_edge_id
                )

                out = torch.ops.pyg.neighbor_sample(*args)
                row, col, node, edge, batch = out[:4] + (None, )

                num_sampled_nodes = num_sampled_edges = None
                if len(out) >= 6:
                    num_sampled_nodes, num_sampled_edges = out[4:6]

                if self.disjoint:
                    batch, node = node.t().contiguous()

            elif torch_geometric.typing.WITH_TORCH_SPARSE:
                if self.disjoint:
                    raise ValueError("'disjoint' sampling not supported for "
                                     "neighbor sampling via 'torch-sparse'. "
                                     "Please install 'pyg-lib' for improved "
                                     "and optimized sampling routines.")

                out = torch.ops.torch_sparse.neighbor_sample(
                    self.colptr,
                    self.row,
                    seed,  # seed
                    self.num_neighbors.get_mapped_values(),
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                )
                node, row, col, edge, batch = out + (None, )
                num_sampled_nodes = num_sampled_edges = None

            else:
                raise ImportError(f"'{self.__class__.__name__}' requires "
                                  f"either 'pyg-lib' or 'torch-sparse'")

            if self.sample_direction == 'backward':
                row, col = col, row

            return SamplerOutput(
                node=node,
                row=row,
                col=col,
                edge=edge,
                batch=batch,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
            )


class BidirectionalNeighborSampler(NeighborSampler):
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: NumNeighborsType,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        replace: bool = False,
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        is_sorted: bool = False,
        share_memory: bool = False,
        directed: bool = True,
    ):

        if isinstance(num_neighbors, NumNeighbors) and isinstance(
                num_neighbors.values, dict) or isinstance(num_neighbors, dict):
            raise RuntimeError(
                "BidirectionalNeighborSampler does not yet support edge "
                "delimited sampling.")

        self.forward_sampler = NeighborSampler(
            data, num_neighbors, subgraph_type, replace, disjoint,
            temporal_strategy, time_attr, weight_attr, is_sorted, share_memory,
            sample_direction='forward', directed=directed)
        self.backward_sampler = NeighborSampler(
            data, num_neighbors, subgraph_type, replace, disjoint,
            temporal_strategy, time_attr, weight_attr, is_sorted, share_memory,
            sample_direction='backward', directed=directed)

        self.num_neighbors = num_neighbors
        self.subgraph_type = subgraph_type

    @property
    def num_neighbors(self) -> NumNeighbors:
        pass

    @num_neighbors.setter
    def num_neighbors(self, num_neighbors: NumNeighborsType):
        pass

    @property
    def is_hetero(self) -> bool:
        pass

    @property
    def is_temporal(self) -> bool:
        pass

    @property
    def disjoint(self) -> bool:
        pass

    @disjoint.setter
    def disjoint(self, disjoint: bool):
        pass

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        return super().sample_from_nodes(inputs)

    def sample_from_edges(
        self,
        inputs: EdgeSamplerInput,
        neg_sampling: Optional[NegativeSampling] = None,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        pass

    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        pass

    def _sample(
        self,
        seed: Union[Tensor, Dict[NodeType, Tensor]],
        seed_time: Optional[Union[Tensor, Dict[NodeType, Tensor]]] = None,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:

        if seed_time is not None:
            raise NotImplementedError(
                "BidirectionalNeighborSampler does not yet support "
                "temporal sampling.")

        if self.is_hetero:
            raise NotImplementedError(
                "BidirectionalNeighborSampler does not yet support "
                "heterogeneous sampling.")
        else:
            current_seed = seed
            current_seed_batch = None
            current_seed_time = seed_time
            seen_seed_set = {int(node) for node in current_seed}
            if self.disjoint:
                current_seed_batch = torch.arange(len(current_seed))
                seen_seed_set = {
                    (int(node), int(batch))
                    for node, batch in zip(current_seed, current_seed_batch)
                }

            iter_results = []

            for n_neighbors in self.num_neighbors.values:
                current_n_neighbors = [n_neighbors]
                self.forward_sampler.num_neighbors = current_n_neighbors
                self.backward_sampler.num_neighbors = current_n_neighbors

                fwd_result = self.forward_sampler._sample(
                    current_seed, current_seed_time, **kwargs)
                bwd_result = self.backward_sampler._sample(
                    current_seed, current_seed_time, **kwargs)
                iter_result = fwd_result.merge_with(bwd_result)
                iter_results.append(iter_result)

                if self.disjoint:
                    iter_seed_global_batch = global_to_local_node_idx(
                        current_seed_batch, iter_result.batch)
                    iter_result.seed_node = seed[iter_seed_global_batch]

                    keep_mask = torch.tensor([
                        (int(node), int(batch)) not in seen_seed_set
                        for node, batch in zip(iter_result.node,
                                               iter_seed_global_batch)
                    ])
                    next_seed = [(int(node), int(batch))
                                 for node, batch in zip(
                                     iter_result.node[keep_mask],
                                     iter_seed_global_batch[keep_mask])
                                 ] if keep_mask.any() else []
                    current_seed, current_seed_batch = torch.tensor(
                        next_seed).reshape(-1, 2).transpose(0, 1).contiguous()
                else:
                    keep_mask = torch.tensor([
                        int(node) not in seen_seed_set
                        for node in iter_result.node
                    ])
                    next_seed = [
                        int(node) for node in iter_result.node[keep_mask]
                    ] if keep_mask.any() else []
                    current_seed = torch.tensor(next_seed)

                seen_seed_set |= set(next_seed)


            return SamplerOutput.collate(iter_results)




def node_sample(
    inputs: NodeSamplerInput,
    sample_fn: Callable,
) -> Union[SamplerOutput, HeteroSamplerOutput]:
    r"""Performs sampling from a :class:`NodeSamplerInput`, leveraging a
    sampling function that accepts a seed and (optionally) a seed time as
    input. Returns the output of this sampling procedure.
    """
    if inputs.input_type is not None:  # Heterogeneous sampling:
        seed = {inputs.input_type: inputs.node}
        seed_time = None
        if inputs.time is not None:
            seed_time = {inputs.input_type: inputs.time}
    else:  # Homogeneous sampling:
        seed = inputs.node
        seed_time = inputs.time

    out = sample_fn(seed, seed_time)
    out.metadata = (inputs.input_id, inputs.time)

    return out


def edge_sample(
    inputs: EdgeSamplerInput,
    sample_fn: Callable,
    num_nodes: Union[int, Dict[NodeType, int]],
    disjoint: bool,
    node_time: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
    neg_sampling: Optional[NegativeSampling] = None,
) -> Union[SamplerOutput, HeteroSamplerOutput]:
    r"""Performs sampling from an edge sampler input, leveraging a sampling
    function of the same signature as `node_sample`.
    """
    input_id = inputs.input_id
    src = inputs.row
    dst = inputs.col
    edge_label = inputs.label
    edge_label_time = inputs.time
    input_type = inputs.input_type

    src_time = dst_time = edge_label_time
    assert edge_label_time is None or disjoint

    assert isinstance(num_nodes, (dict, int))
    if not isinstance(num_nodes, dict):
        num_src_nodes = num_dst_nodes = num_nodes
    else:
        num_src_nodes = num_nodes[input_type[0]]
        num_dst_nodes = num_nodes[input_type[-1]]

    num_pos = src.numel()
    num_neg = 0


    if neg_sampling is not None:
        num_neg = math.ceil(num_pos * neg_sampling.amount)

        if neg_sampling.is_binary():
            if isinstance(node_time, dict):
                src_node_time = node_time.get(input_type[0])
            else:
                src_node_time = node_time

            src_neg = neg_sample(src, neg_sampling, num_src_nodes, src_time,
                                 src_node_time, endpoint='src')
            src = torch.cat([src, src_neg], dim=0)

            if isinstance(node_time, dict):
                dst_node_time = node_time.get(input_type[-1])
            else:
                dst_node_time = node_time

            dst_neg = neg_sample(dst, neg_sampling, num_dst_nodes, dst_time,
                                 dst_node_time, endpoint='dst')
            dst = torch.cat([dst, dst_neg], dim=0)

            if edge_label is None:
                edge_label = torch.ones(num_pos)
            size = (num_neg, ) + edge_label.size()[1:]
            edge_neg_label = edge_label.new_zeros(size)
            edge_label = torch.cat([edge_label, edge_neg_label])

            if edge_label_time is not None:
                src_time = dst_time = edge_label_time.repeat(
                    1 + math.ceil(neg_sampling.amount))[:num_pos + num_neg]

        elif neg_sampling.is_triplet():
            if isinstance(node_time, dict):
                dst_node_time = node_time.get(input_type[-1])
            else:
                dst_node_time = node_time

            dst_neg = neg_sample(dst, neg_sampling, num_dst_nodes, dst_time,
                                 dst_node_time, endpoint='dst')
            dst = torch.cat([dst, dst_neg], dim=0)

            assert edge_label is None

            if edge_label_time is not None:
                dst_time = edge_label_time.repeat(1 + neg_sampling.amount)


    if input_type is not None:
        seed_time_dict = None
        if input_type[0] != input_type[-1]:  # Two distinct node types:

            if not disjoint:
                src, inverse_src = src.unique(return_inverse=True)
                dst, inverse_dst = dst.unique(return_inverse=True)

            seed_dict = {input_type[0]: src, input_type[-1]: dst}

            if edge_label_time is not None:  # Always disjoint.
                seed_time_dict = {
                    input_type[0]: src_time,
                    input_type[-1]: dst_time,
                }

        else:  # Only a single node type: Merge both source and destination.

            seed = torch.cat([src, dst], dim=0)

            if not disjoint:
                seed, inverse_seed = seed.unique(return_inverse=True)

            seed_dict = {input_type[0]: seed}

            if edge_label_time is not None:  # Always disjoint.
                seed_time_dict = {
                    input_type[0]: torch.cat([src_time, dst_time], dim=0),
                }

        out = sample_fn(seed_dict, seed_time_dict)

        if disjoint:
            for key, batch in out.batch.items():
                out.batch[key] = batch % num_pos

        if neg_sampling is None or neg_sampling.is_binary():
            if disjoint:
                if input_type[0] != input_type[-1]:
                    edge_label_index = torch.arange(num_pos + num_neg)
                    edge_label_index = edge_label_index.repeat(2).view(2, -1)
                else:
                    edge_label_index = torch.arange(2 * (num_pos + num_neg))
                    edge_label_index = edge_label_index.view(2, -1)
            else:
                if input_type[0] != input_type[-1]:
                    edge_label_index = torch.stack([
                        inverse_src,
                        inverse_dst,
                    ], dim=0)
                else:
                    edge_label_index = inverse_seed.view(2, -1)

            out.metadata = (input_id, edge_label_index, edge_label, src_time)

        elif neg_sampling.is_triplet():
            if disjoint:
                src_index = torch.arange(num_pos)
                if input_type[0] != input_type[-1]:
                    dst_pos_index = torch.arange(num_pos)
                    dst_neg_index = torch.arange(
                        num_pos, seed_dict[input_type[-1]].numel())
                    dst_neg_index = dst_neg_index.view(-1, num_pos).t()
                else:
                    dst_pos_index = torch.arange(num_pos, 2 * num_pos)
                    dst_neg_index = torch.arange(
                        2 * num_pos, seed_dict[input_type[-1]].numel())
                    dst_neg_index = dst_neg_index.view(-1, num_pos).t()
            else:
                if input_type[0] != input_type[-1]:
                    src_index = inverse_src
                    dst_pos_index = inverse_dst[:num_pos]
                    dst_neg_index = inverse_dst[num_pos:]
                else:
                    src_index = inverse_seed[:num_pos]
                    dst_pos_index = inverse_seed[num_pos:2 * num_pos]
                    dst_neg_index = inverse_seed[2 * num_pos:]

            dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)

            out.metadata = (
                input_id,
                src_index,
                dst_pos_index,
                dst_neg_index,
                src_time,
            )


    else:

        seed = torch.cat([src, dst], dim=0)
        seed_time = None

        if not disjoint:
            seed, inverse_seed = seed.unique(return_inverse=True)

        if edge_label_time is not None:  # Always disjoint.
            seed_time = torch.cat([src_time, dst_time])

        out = sample_fn(seed, seed_time)

        if neg_sampling is None or neg_sampling.is_binary():
            if disjoint:
                out.batch = out.batch % num_pos
                edge_label_index = torch.arange(seed.numel()).view(2, -1)
            else:
                edge_label_index = inverse_seed.view(2, -1)

            out.metadata = (input_id, edge_label_index, edge_label, src_time)

        elif neg_sampling.is_triplet():
            if disjoint:
                out.batch = out.batch % num_pos
                src_index = torch.arange(num_pos)
                dst_pos_index = torch.arange(num_pos, 2 * num_pos)
                dst_neg_index = torch.arange(2 * num_pos, seed.numel())
                dst_neg_index = dst_neg_index.view(-1, num_pos).t()
            else:
                src_index = inverse_seed[:num_pos]
                dst_pos_index = inverse_seed[num_pos:2 * num_pos]
                dst_neg_index = inverse_seed[2 * num_pos:]
            dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)

            out.metadata = (
                input_id,
                src_index,
                dst_pos_index,
                dst_neg_index,
                src_time,
            )

    return out


def neg_sample(
    seed: Tensor,
    neg_sampling: NegativeSampling,
    num_nodes: int,
    seed_time: Optional[Tensor],
    node_time: Optional[Tensor],
    endpoint: Literal['str', 'dst'],
) -> Tensor:
    num_neg = math.ceil(seed.numel() * neg_sampling.amount)

    if node_time is None:
        return neg_sampling.sample(num_neg, endpoint, num_nodes)

    assert seed_time is not None
    num_samples = math.ceil(neg_sampling.amount)
    seed_time = seed_time.view(1, -1).expand(num_samples, -1)

    out = neg_sampling.sample(num_samples * seed.numel(), endpoint, num_nodes)
    out = out.view(num_samples, seed.numel())
    mask = node_time[out] > seed_time  # holds all invalid samples.
    neg_sampling_complete = False
    for _ in range(5):  # pragma: no cover
        num_invalid = int(mask.sum())
        if num_invalid == 0:
            neg_sampling_complete = True
            break

        out[mask] = tmp = neg_sampling.sample(num_invalid, endpoint, num_nodes)
        mask[mask.clone()] = node_time[tmp] >= seed_time[mask]

    if not neg_sampling_complete:  # pragma: no cover
        out[mask] = node_time.argmin()

    return out.view(-1)[:num_neg]
