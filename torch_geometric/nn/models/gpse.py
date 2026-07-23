import logging
import os
import os.path as osp
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from tqdm import trange

import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import (
    ResGatedGraphConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_batch


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
    ) -> None:
        super().__init__()
        self.model = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class ResGatedGCNConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = ResGatedGraphConv(
            in_channels,
            out_channels,
            bias=bias,
            **kwargs,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class GeneralLayer(torch.nn.Module):
    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        has_batch_norm: bool,
        has_l2_norm: bool,
        dropout: float,
        act: Optional[str],
        **kwargs,
    ):
        super().__init__()
        self.has_l2_norm = has_l2_norm

        layer_dict = {
            'linear': Linear,
            'resgatedgcnconv': ResGatedGCNConv,
        }
        self.layer = layer_dict[name](
            in_channels,
            out_channels,
            bias=not has_batch_norm,
            **kwargs,
        )
        post_layers = []
        if has_batch_norm:
            post_layers.append(
                torch.nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.1))
        if dropout > 0:
            post_layers.append(torch.nn.Dropout(p=dropout, inplace=False))
        if act is not None:
            post_layers.append(activation_resolver(act))
        self.post_layer = nn.Sequential(*post_layers)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2_norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2_norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


class GeneralMultiLayer(torch.nn.Module):
    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int],
        num_layers: int,
        has_batch_norm: bool,
        has_l2_norm: bool,
        dropout: float,
        act: str,
        final_act: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or out_channels

        for i in range(num_layers):
            d_in = in_channels if i == 0 else hidden_channels
            d_out = out_channels if i == num_layers - 1 else hidden_channels
            layer = GeneralLayer(
                name=name,
                in_channels=d_in,
                out_channels=d_out,
                has_batch_norm=has_batch_norm,
                has_l2_norm=has_l2_norm,
                dropout=dropout,
                act=None if i == num_layers - 1 and not final_act else act,
                **kwargs,
            )
            self.add_module(f'Layer_{i}', layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


class BatchNorm1dNode(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(channels, eps=1e-5, momentum=0.1)

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(channels, eps=1e-5, momentum=0.1)

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int],
        num_layers: int,
        has_batch_norm: bool = True,
        has_l2_norm: bool = True,
        dropout: float = 0.2,
        act: str = 'relu',
        **kwargs,
    ):
        super().__init__()
        hidden_channels = hidden_channels or in_channels

        layers = []
        if num_layers > 1:
            layer = GeneralMultiLayer(
                'linear',
                in_channels,
                hidden_channels,
                hidden_channels,
                num_layers - 1,
                has_batch_norm,
                has_l2_norm,
                dropout,
                act,
                final_act=True,
                **kwargs,
            )
            layers.append(layer)
        layers.append(Linear(hidden_channels, out_channels, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class GNNStackStage(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        layer_type: str,
        stage_type: str = 'skipsum',
        final_l2_norm: bool = True,
        has_batch_norm: bool = True,
        has_l2_norm: bool = True,
        dropout: float = 0.2,
        act: Optional[str] = 'relu',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.stage_type = stage_type
        self.final_l2_norm = final_l2_norm

        for i in range(num_layers):
            if stage_type == 'skipconcat':
                if i == 0:
                    d_in = in_channels
                else:
                    d_in = in_channels + i * out_channels
            else:
                d_in = in_channels if i == 0 else out_channels
            layer = GeneralLayer(layer_type, d_in, out_channels,
                                 has_batch_norm, has_l2_norm, dropout, act)
            self.add_module(f'layer{i}', layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if self.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif self.stage_type == 'skipconcat' and i < self.num_layers - 1:
                batch.x = torch.cat([x, batch.x], dim=1)

        if self.final_l2_norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)

        return batch


class GNNInductiveHybridMultiHead(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_node_targets: int,
        num_graph_targets: int,
        layers_post_mp: int,
        virtual_node: bool = True,
        multi_head_dim_inner: int = 32,
        graph_pooling: str = 'add',
        has_bn: bool = True,
        has_l2norm: bool = True,
        dropout: float = 0.2,
        act: str = 'relu',
    ):
        super().__init__()
        pool_dict = {
            'add': global_add_pool,
            'max': global_max_pool,
            'mean': global_mean_pool
        }
        self.node_target_dim = num_node_targets
        self.graph_target_dim = num_graph_targets
        self.virtual_node = virtual_node
        num_layers = layers_post_mp

        self.node_post_mps = nn.ModuleList([
            MLP(dim_in, 1, multi_head_dim_inner, num_layers, has_bn,
                has_l2norm, dropout, act) for _ in range(self.node_target_dim)
        ])

        self.graph_pooling = pool_dict[graph_pooling]

        self.graph_post_mp = MLP(dim_in, self.graph_target_dim, dim_in,
                                 num_layers, has_bn, has_l2norm, dropout, act)

    def _pad_and_stack(self, x1: torch.Tensor, x2: torch.Tensor, pad1: int,
                       pad2: int):
        padded_x1 = nn.functional.pad(x1, (0, pad2))
        padded_x2 = nn.functional.pad(x2, (pad1, 0))
        return torch.vstack([padded_x1, padded_x2])

    def _apply_index(self, batch, virtual_node: bool, pad_node: int,
                     pad_graph: int):
        graph_pred, graph_true = batch.graph_feature, batch.y_graph
        node_pred, node_true = batch.node_feature, batch.y
        if virtual_node:
            idx = torch.concat([
                torch.where(batch.batch == i)[0][:-1]
                for i in range(batch.batch.max().item() + 1)
            ])
            node_pred, node_true = node_pred[idx], node_true[idx]

        pred = self._pad_and_stack(node_pred, graph_pred, pad_node, pad_graph)
        true = self._pad_and_stack(node_true, graph_true, pad_node, pad_graph)

        return pred, true

    def forward(self, batch):
        batch.node_feature = torch.hstack(
            [m(batch.x) for m in self.node_post_mps])
        graph_emb = self.graph_pooling(batch.x, batch.batch)
        batch.graph_feature = self.graph_post_mp(graph_emb)
        return self._apply_index(batch, self.virtual_node,
                                 self.node_target_dim, self.graph_target_dim)


class IdentityHead(torch.nn.Module):
    def forward(self, batch):
        return batch.x, batch.y


class GPSE(torch.nn.Module):

    url_dict = {
        'molpcba':
        'https://zenodo.org/record/8145095/files/'
        'gpse_model_molpcba_1.0.pt',
        'zinc':
        'https://zenodo.org/record/8145095/files/gpse_model_zinc_1.0.pt',
        'pcqm4mv2':
        'https://zenodo.org/record/8145095/files/'
        'gpse_model_pcqm4mv2_1.0.pt',
        'geom':
        'https://zenodo.org/record/8145095/files/gpse_model_geom_1.0.pt',
        'chembl':
        'https://zenodo.org/record/8145095/files/gpse_model_chembl_1.0.pt'
    }

    def __init__(
        self,
        dim_in: int = 20,
        dim_out: int = 51,
        dim_inner: int = 512,
        layer_type: str = 'resgatedgcnconv',
        layers_pre_mp: int = 1,
        layers_mp: int = 20,
        layers_post_mp: int = 2,
        num_node_targets: int = 51,
        num_graph_targets: int = 11,
        stage_type: str = 'skipsum',
        has_bn: bool = True,
        head_bn: bool = False,
        final_l2norm: bool = True,
        has_l2norm: bool = True,
        dropout: float = 0.2,
        has_act: bool = True,
        final_act: bool = True,
        act: str = 'relu',
        virtual_node: bool = True,
        multi_head_dim_inner: int = 32,
        graph_pooling: str = 'add',
        use_repr: bool = True,
        repr_type: str = 'no_post_mp',
        bernoulli_threshold: float = 0.5,
    ):
        super().__init__()

        self.use_repr = use_repr
        self.repr_type = repr_type
        self.bernoulli_threshold = bernoulli_threshold

        if layers_pre_mp > 0:
            self.pre_mp = GeneralMultiLayer(
                name='linear',
                in_channels=dim_in,
                out_channels=dim_inner,
                hidden_channels=dim_inner,
                num_layers=layers_pre_mp,
                has_batch_norm=has_bn,
                has_l2_norm=has_l2norm,
                dropout=dropout,
                act=act,
                final_act=final_act,
            )
            dim_in = dim_inner
        if layers_mp > 0:
            self.mp = GNNStackStage(
                in_channels=dim_in,
                out_channels=dim_inner,
                num_layers=layers_mp,
                layer_type=layer_type,
                stage_type=stage_type,
                final_l2_norm=final_l2norm,
                has_batch_norm=has_bn,
                has_l2_norm=has_l2norm,
                dropout=dropout,
                act=act if has_act else None,
            )

        self.post_mp = GNNInductiveHybridMultiHead(
            dim_inner,
            dim_out,
            num_node_targets,
            num_graph_targets,
            layers_post_mp,
            virtual_node,
            multi_head_dim_inner,
            graph_pooling,
            head_bn,
            has_l2norm,
            dropout,
            act,
        )

        self.reset_parameters()

    def reset_parameters(self):
        pass

    @classmethod
    def from_pretrained(cls, name: str, root: str = 'GPSE_pretrained'):
        r"""Returns a pretrained :class:`GPSE` model on a dataset.

        Args:
            name (str): The name of the dataset (:obj:`"molpcba"`,
                :obj:`"zinc"`, :obj:`"pcqm4mv2"`, :obj:`"geom"`,
                :obj:`"chembl"`).
            root (str, optional): The root directory to save the pre-trained
                model. (default: :obj:`"GPSE_pretrained"`)
        """
        root = osp.expanduser(osp.normpath(root))
        os.makedirs(root, exist_ok=True)
        path = download_url(cls.url_dict[name], root)

        model = GPSE()  # All pretrained models use the default arguments
        model_state = torch.load(path, map_location='cpu')['model_state']
        model_state_new = OrderedDict([(k.split('.', 1)[1], v)
                                       for k, v in model_state.items()])
        model.load_state_dict(model_state_new)

        if model.use_repr:
            if model.repr_type == 'one_layer_before':
                model.post_mp.layer_post_mp.model[-1] = torch.nn.Identity()
            elif model.repr_type == 'no_post_mp':
                model.post_mp = IdentityHead()
            else:
                raise ValueError(f"Unknown type '{model.repr_type}'")

        model.eval()
        return model

    def forward(self, batch):
        batch = batch.clone()
        for module in self.children():
            batch = module(batch)
        return batch


class GPSENodeEncoder(torch.nn.Module):
    def __init__(self, dim_emb: int, dim_pe_in: int, dim_pe_out: int,
                 dim_in: int = None, expand_x=False, norm_type='batchnorm',
                 model_type='mlp', n_layers=2, dropout_be=0.5, dropout_ae=0.2):
        super().__init__()

        assert dim_emb > dim_pe_out, ('Desired GPSE dimension (dim_pe_out) '
                                      'must be smaller than the final node '
                                      'embedding dimension (dim_emb).')

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe_out)
        self.expand_x = expand_x

        self.raw_norm = None
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(dim_pe_in)

        self.dropout_be = nn.Dropout(p=dropout_be)
        self.dropout_ae = nn.Dropout(p=dropout_ae)

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(torch.nn.Linear(dim_pe_in, dim_pe_out))
                layers.append(activation())
            else:
                layers.append(torch.nn.Linear(dim_pe_in, 2 * dim_pe_out))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(
                        torch.nn.Linear(2 * dim_pe_out, 2 * dim_pe_out))
                    layers.append(activation())
                layers.append(torch.nn.Linear(2 * dim_pe_out, dim_pe_out))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(dim_pe_in, dim_pe_out)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, x, pos_enc):
        pos_enc = self.dropout_be(pos_enc)
        pos_enc = self.raw_norm(pos_enc) if self.raw_norm else pos_enc
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe
        pos_enc = self.dropout_ae(pos_enc)

        h = self.linear_x(x) if self.expand_x else x

        return torch.cat((h, pos_enc), 1)


@torch.no_grad()
def gpse_process(
    model: Module,
    data: Data,
    rand_type: str,
    use_vn: bool = True,
    bernoulli_thresh: float = 0.5,
    neighbor_loader: bool = False,
    num_neighbors: Optional[List[int]] = None,
    fillval: int = 5,
    layers_mp: int = None,
    **kwargs,
) -> torch.Tensor:
    r"""Processes the data using the :class:`GPSE` model to generate and append
    GPSE encodings. Identical to :obj:`gpse_process_batch`, but operates on a
    single :class:`~torch_geometric.data.Dataset` object.

    Unlike transform-based GPSE processing (i.e.
    :class:`~torch_geometric.transforms.AddGPSE`), the :obj:`use_vn` argument
    does not append virtual nodes if set to :obj:`True`, and instead assumes
    the input graphs to :obj:`gpse_process` already have virtual nodes. Under
    normal circumstances, one does not need to call this function; running
    :obj:`precompute_GPSE` on your whole dataset is advised instead.

    Args:
        model (Module): The :class:`GPSE` model.
        data (torch_geometric.data.Data): A :class:`~torch_geometric.data.Data`
            object.
        rand_type (str, optional): Type of random features to use. Options are
            :obj:`NormalSE`, :obj:`UniformSE`, :obj:`BernoulliSE`.
            (default: :obj:`NormalSE`)
        use_vn (bool, optional): Whether the input graphs have virtual nodes.
            (default: :obj:`True`)
        bernoulli_thresh (float, optional): Threshold for Bernoulli sampling of
            virtual nodes. (default: :obj:`0.5`)
        neighbor_loader (bool, optional): Whether to use :obj:`NeighborLoader`.
            (default: :obj:`False`)
        num_neighbors (List[int], optional): Number of neighbors to consider
            for each message-passing layer. (default: :obj:`[30, 20, 10]`)
        fillval (int, optional): Value to fill for missing
            :obj:`num_neighbors`. (default: :obj:`5`)
        layers_mp (int, optional): Number of message-passing layers.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments for :obj:`NeighborLoader`.

    Returns:
        torch.Tensor: A tensor corresponding to the original
        :class:`~torch_geometric.data.Data` object, with :class:`GPSE`
        encodings appended as :obj:`out.pestat_GPSE` attribute.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = data.num_nodes
    dim_in = model.state_dict()[list(model.state_dict())[0]].shape[1]

    if rand_type == 'NormalSE':
        rand = np.random.normal(loc=0, scale=1.0, size=(n, dim_in))
    elif rand_type == 'UniformSE':
        rand = np.random.uniform(low=0.0, high=1.0, size=(n, dim_in))
    elif rand_type == 'BernoulliSE':
        rand = np.random.uniform(low=0.0, high=1.0, size=(n, dim_in))
        rand = (rand < bernoulli_thresh)
    else:
        raise ValueError(f'Unknown {rand_type=!r}')
    data.x = torch.from_numpy(rand.astype('float32'))

    if use_vn:
        data.x[-1] = 0

    model, data = model.to(device), data.to(device)
    # Generate encodings using the pretrained encoder
    if neighbor_loader:
        if layers_mp is None:
            raise ValueError('Please provide the number of message-passing '
                             'layers as "layers_mp".')

        num_neighbors = num_neighbors or [30, 20, 10]
        diff = layers_mp - len(num_neighbors)
        if fillval > 0 and diff > 0:
            num_neighbors += [fillval] * diff

        loader = NeighborLoader(data, num_neighbors=num_neighbors,
                                shuffle=False, pin_memory=True, **kwargs)
        out_list = []
        pbar = trange(data.num_nodes, position=2)
        for batch in loader:
            out, _ = model(batch.to(device))
            out = out[:batch.batch_size].to("cpu", non_blocking=True)
            out_list.append(out)
            pbar.update(batch.batch_size)
        out = torch.vstack(out_list)
    else:
        out, _ = model(data)
        out = out.to("cpu")

    return out


@torch.no_grad()
def gpse_process_batch(
    model: GPSE,
    batch,
    rand_type: str,
    use_vn: bool = True,
    bernoulli_thresh: float = 0.5,
    neighbor_loader: bool = False,
    num_neighbors: Optional[List[int]] = None,
    fillval: int = 5,
    layers_mp: int = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass


@torch.no_grad()
def precompute_GPSE(model: GPSE, dataset: Dataset, use_vn: bool = True,
                    rand_type: str = 'NormalSE', **kwargs):
    pass


def cosim_col_sep(pred: torch.Tensor, true: torch.Tensor,
                  batch_idx: torch.Tensor) -> torch.Tensor:
    r"""Calculates the average cosine similarity between predicted and true
    features on a batch of graphs.

    Args:
        pred (torch.Tensor): Predicted outputs.
        true (torch.Tensor): Value of ground truths.
        batch_idx (torch.Tensor): Batch indices to separate the graphs.

    Returns:
        torch.Tensor: Average cosine similarity per graph in batch.

    Raises:
        ValueError: If batch_index is not specified.
    """
    if batch_idx is None:
        raise ValueError("mae_cosim_col_sep requires batch index as "
                         "input to distinguish different graphs.")
    batch_idx = batch_idx + 1 if batch_idx.min() == -1 else batch_idx
    pred_dense = to_dense_batch(pred, batch_idx)[0]
    true_dense = to_dense_batch(true, batch_idx)[0]
    mask = (true_dense == 0).all(1)  # exclude trivial features from loss
    loss = 1 - F.cosine_similarity(pred_dense, true_dense, dim=1)[~mask].mean()
    return loss


def gpse_loss(pred: torch.Tensor, true: torch.Tensor,
              batch_idx: torch.Tensor = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculates :class:`GPSE` loss as the sum of MAE loss and cosine
    similarity loss over a batch of graphs.

    Args:
        pred (torch.Tensor): Predicted outputs.
        true (torch.Tensor): Value of ground truths.
        batch_idx (torch.Tensor): Batch indices to separate the graphs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A two-tuple of tensors corresponding
        to the :class:`GPSE` loss and the predicted node-and-graph level
        outputs.

    """
    if batch_idx is None:
        raise ValueError("mae_cosim_col_sep requires batch index as "
                         "input to distinguish different graphs.")
    mae_loss = F.l1_loss(pred, true)
    cosim_loss = cosim_col_sep(pred, true, batch_idx)
    loss = mae_loss + cosim_loss
    return loss, pred


def process_batch_idx(batch_idx, true, use_vn=True):
    r"""Processes batch indices to adjust for the removal of virtual nodes, and
    pads batch index for hybrid tasks.

    Args:
        batch_idx: Batch indices to separate the graphs.
        true: Value of ground truths.
        use_vn: If input graphs have virtual nodes that need to be removed.

    Returns:
        torch.Tensor: Batch indices that separate the graphs.
    """
    if batch_idx is None:
        return
    if use_vn:  # remove virtual node
        batch_idx = torch.concat([
            batch_idx[batch_idx == i][:-1]
            for i in range(batch_idx.max().item() + 1)
        ])
    if (pad := true.shape[0] - batch_idx.shape[0]) > 0:
        pad_idx = -torch.ones(pad, dtype=torch.long, device=batch_idx.device)
        batch_idx = torch.hstack([batch_idx, pad_idx])
    return batch_idx
