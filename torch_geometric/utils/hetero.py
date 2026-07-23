from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor
from torch.nn import ParameterDict

from torch_geometric.typing import Adj, EdgeType, NodeType, SparseTensor
from torch_geometric.utils import is_sparse, to_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes_dict


def group_hetero_graph(
    edge_index_dict: Dict[EdgeType, Tensor],
    num_nodes_dict: Optional[Dict[NodeType, int]] = None,
) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Dict[Union[str, int], Tensor],
        Dict[Union[NodeType, EdgeType], int],
]:
    pass


def get_unused_node_types(node_types: List[NodeType],
                          edge_types: List[EdgeType]) -> Set[NodeType]:
    dst_node_types = {edge_type[-1] for edge_type in edge_types}
    return set(node_types) - set(dst_node_types)


def check_add_self_loops(
    module: torch.nn.Module,
    edge_types: List[EdgeType],
) -> None:
    is_bipartite = any([key[0] != key[-1] for key in edge_types])
    if is_bipartite and getattr(module, 'add_self_loops', False):
        raise ValueError(
            f"'add_self_loops' attribute set to 'True' on module '{module}' "
            f"for use with edge type(s) '{edge_types}'. This will lead to "
            f"incorrect message passing results.")


def construct_bipartite_edge_index(
    edge_index_dict: Dict[EdgeType, Adj],
    src_offset_dict: Dict[EdgeType, int],
    dst_offset_dict: Dict[NodeType, int],
    edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Adj, Optional[Tensor]]:
    """Constructs a tensor of edge indices by concatenating edge indices
    for each edge type. The edge indices are increased by the offset of the
    source and destination nodes.

    Args:
        edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
            dictionary holding graph connectivity information for each
            individual edge type, either as a :class:`torch.Tensor` of
            shape :obj:`[2, num_edges]` or a
            :class:`torch_sparse.SparseTensor`.
        src_offset_dict (Dict[Tuple[str, str, str], int]): A dictionary of
            offsets to apply to the source node type for each edge type.
        dst_offset_dict (Dict[str, int]): A dictionary of offsets to apply for
            destination node types.
        edge_attr_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
            dictionary holding edge features for each individual edge type.
            (default: :obj:`None`)
        num_nodes (int, optional): The final number of nodes in the bipartite
            adjacency matrix. (default: :obj:`None`)
    """
    is_sparse_tensor = False
    edge_indices: List[Tensor] = []
    edge_attrs: List[Tensor] = []
    for edge_type, src_offset in src_offset_dict.items():
        edge_index = edge_index_dict[edge_type]
        dst_offset = dst_offset_dict[edge_type[-1]]

        is_sparse_tensor = isinstance(edge_index, SparseTensor)
        if is_sparse(edge_index):
            edge_index, _ = to_edge_index(edge_index)
            edge_index = edge_index.flip([0])
        else:
            edge_index = edge_index.clone()

        edge_index[0] += src_offset
        edge_index[1] += dst_offset
        edge_indices.append(edge_index)

        if edge_attr_dict is not None:
            if isinstance(edge_attr_dict, ParameterDict):
                value = edge_attr_dict['__'.join(edge_type)]
            else:
                value = edge_attr_dict[edge_type]
            if value.size(0) != edge_index.size(1):
                value = value.expand(edge_index.size(1), -1)
            edge_attrs.append(value)

    edge_index = torch.cat(edge_indices, dim=1)

    edge_attr: Optional[Tensor] = None
    if edge_attr_dict is not None:
        edge_attr = torch.cat(edge_attrs, dim=0)

    if is_sparse_tensor:
        edge_index = SparseTensor(
            row=edge_index[1],
            col=edge_index[0],
            value=edge_attr,
            sparse_sizes=(num_nodes, num_nodes),
        )

    return edge_index, edge_attr
