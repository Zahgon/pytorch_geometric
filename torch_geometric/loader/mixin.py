import glob
import logging
import os
import os.path as osp
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
import torch

from torch_geometric.data import HeteroData


def get_numa_nodes_cores() -> Dict[str, Any]:
    pass


class WorkerInitWrapper:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, worker_id: int) -> None:
        if self.func is not None:
            self.func(worker_id)


class LogMemoryMixin:
    def _mem_init_fn(self, worker_id: int) -> None:
        pass

    @contextmanager
    def enable_memory_log(self) -> None:
        pass


class MultithreadingMixin:
    def _mt_init_fn(self, worker_id: int) -> None:
        pass

    @contextmanager
    def enable_multithreading(
        self,
        worker_threads: Optional[int] = None,
    ) -> None:
        pass


class AffinityMixin:
    def _aff_init_fn(self, worker_id: int) -> None:
        pass

    @contextmanager
    def enable_cpu_affinity(
        self,
        loader_cores: Optional[Union[List[List[int]], List[int]]] = None,
    ) -> None:
        pass
