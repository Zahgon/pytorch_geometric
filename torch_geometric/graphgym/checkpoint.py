import glob
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Union

import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.io import fs

MODEL_STATE = 'model_state'
OPTIMIZER_STATE = 'optimizer_state'
SCHEDULER_STATE = 'scheduler_state'


def load_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = -1,
) -> int:
    pass


def save_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
):
    pass


def remove_ckpt(epoch: int = -1):
    pass


def clean_ckpt():
    pass




def get_ckpt_dir() -> str:
    return osp.join(cfg.run_dir, 'ckpt')


def get_ckpt_path(epoch: Union[int, str]) -> str:
    pass


def get_ckpt_epochs() -> List[int]:
    pass


def get_ckpt_epoch(epoch: int) -> int:
    pass
