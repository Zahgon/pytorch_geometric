import logging
import math
import os
import sys
import time
from typing import Any, Dict, Optional

import torch

from torch_geometric.graphgym import register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import Callback, pl
from torch_geometric.graphgym.utils.device import get_current_gpu_usage
from torch_geometric.graphgym.utils.io import dict_to_json, dict_to_tb


def set_printing():
    """Set up printing options."""
    logging.root.handlers = []
    logging_cfg = {'level': logging.INFO, 'format': '%(message)s'}
    os.makedirs(cfg.run_dir, exist_ok=True)
    h_file = logging.FileHandler(f'{cfg.run_dir}/logging.log')
    h_stdout = logging.StreamHandler(sys.stdout)
    if cfg.print == 'file':
        logging_cfg['handlers'] = [h_file]
    elif cfg.print == 'stdout':
        logging_cfg['handlers'] = [h_stdout]
    elif cfg.print == 'both':
        logging_cfg['handlers'] = [h_file, h_stdout]
    else:
        raise ValueError('Print option not supported')
    logging.basicConfig(**logging_cfg)


class Logger:
    def __init__(self, name='train', task_type=None):
        self.name = name
        self.task_type = task_type

        self._epoch_total = cfg.optim.max_epoch
        self._time_total = 0  # won't be reset

        self.out_dir = f'{cfg.run_dir}/{name}'
        os.makedirs(self.out_dir, exist_ok=True)
        if cfg.tensorboard_each_run:
            from tensorboardX import SummaryWriter
            self.tb_writer = SummaryWriter(self.out_dir)

        self.reset()

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def reset(self):
        self._iter = 0
        self._size_current = 0
        self._loss = 0
        self._lr = 0
        self._params = 0
        self._time_used = 0
        self._true = []
        self._pred = []
        self._custom_stats = {}

    def basic(self):
        pass

    def custom(self):
        pass

    def _get_pred_int(self, pred_score):
        pass

    def classification_binary(self):
        pass

    def classification_multi(self):
        pass

    def regression(self):
        pass

    def time_iter(self):
        pass

    def eta(self, epoch_current):
        pass

    def update_stats(self, true, pred, loss, lr, time_used, params, **kwargs):
        pass

    def write_iter(self):
        raise NotImplementedError

    def write_epoch(self, cur_epoch):
        pass

    def close(self):
        if cfg.tensorboard_each_run:
            self.tb_writer.close()


def infer_task():
    num_label = cfg.share.dim_out
    if cfg.dataset.task_type == 'classification':
        if num_label <= 2:
            task_type = 'classification_binary'
        else:
            task_type = 'classification_multi'
    else:
        task_type = cfg.dataset.task_type
    return task_type


def create_logger():
    r"""Create logger for the experiment."""
    loggers = []
    names = ['train', 'val', 'test']
    for i, _ in enumerate(range(cfg.share.num_splits)):
        loggers.append(Logger(name=names[i], task_type=infer_task()))
    return loggers


class LoggerCallback(Callback):
    def __init__(self):
        self._logger = create_logger()
        self._train_epoch_start_time = None
        self._val_epoch_start_time = None
        self._test_epoch_start_time = None

    @property
    def train_logger(self) -> Any:
        pass

    @property
    def val_logger(self) -> Any:
        pass

    @property
    def test_logger(self) -> Any:
        pass

    def close(self):
        for logger in self._logger:
            logger.close()

    def _get_stats(
        self,
        epoch_start_time: int,
        outputs: Dict[str, Any],
        trainer: 'pl.Trainer',
    ) -> Dict:
        pass

    def on_train_epoch_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ):
        pass

    def on_validation_epoch_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ):
        pass

    def on_test_epoch_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ):
        pass

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ):
        pass

    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[Dict[str, Any]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        pass

    def on_test_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[Dict[str, Any]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        pass

    def on_train_epoch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ):
        pass

    def on_validation_epoch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ):
        pass

    def on_test_epoch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ):
        pass

    def on_fit_end(self, trainer, pl_module):
        pass
