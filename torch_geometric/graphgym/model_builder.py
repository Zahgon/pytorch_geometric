import time
from typing import Any, Dict, Tuple

import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import LightningModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import network_dict, register_network

register_network('gnn', GNN)


class GraphGymModule(LightningModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = network_dict[cfg.model.type](dim_in=dim_in,
                                                  dim_out=dim_out)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        pass

    def _shared_step(self, batch, split: str) -> Dict:
        pass

    def training_step(self, batch, *args, **kwargs):
        pass

    def validation_step(self, batch, *args, **kwargs):
        pass

    def test_step(self, batch, *args, **kwargs):
        pass

    @property
    def encoder(self) -> torch.nn.Module:
        return self.model.encoder

    @property
    def mp(self) -> torch.nn.Module:
        pass

    @property
    def post_mp(self) -> torch.nn.Module:
        pass

    @property
    def pre_mp(self) -> torch.nn.Module:
        pass

    def lr_scheduler_step(self, *args, **kwargs):
        pass


def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (bool, optional): Whether to transfer the model to the
            specified device. (default: :obj:`True`)
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    if 'classification' == cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    model = GraphGymModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model
