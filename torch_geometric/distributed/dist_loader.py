import atexit
import logging
import os
from typing import Any, Optional, Union

import torch.distributed
import torch.multiprocessing as mp

from torch_geometric.distributed import DistNeighborSampler
from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed.rpc import (
    global_barrier,
    init_rpc,
    shutdown_rpc,
)
from torch_geometric.loader.base import DataLoaderIterator


class DistLoader:
    def __init__(
        self,
        current_ctx: DistContext,
        master_addr: Optional[str] = None,
        master_port: Optional[Union[int, str]] = None,
        channel: Optional[mp.Queue] = None,
        num_rpc_threads: int = 16,
        rpc_timeout: int = 180,
        dist_sampler: DistNeighborSampler = None,
        **kwargs,
    ):
        if master_addr is None and os.environ.get('MASTER_ADDR') is not None:
            master_addr = os.environ['MASTER_ADDR']
        if master_addr is None:
            raise ValueError(f"Missing master address for RPC communication "
                             f"in '{self.__class__.__name__}'. Try to provide "
                             f"it or set it via the 'MASTER_ADDR' environment "
                             f"variable.")

        if master_port is None and os.environ.get('MASTER_PORT') is not None:
            master_port = int(os.environ['MASTER_PORT']) + 1
        if master_port is None:
            raise ValueError(f"Missing master port for RPC communication in "
                             f"'{self.__class__.__name__}'. Try to provide it "
                             f"or set it via the 'MASTER_ADDR' environment "
                             f"variable.")

        assert num_rpc_threads > 0
        assert rpc_timeout > 0

        self.dist_sampler = dist_sampler
        self.current_ctx = current_ctx
        self.master_addr = master_addr
        self.master_port = master_port
        self.channel = channel
        self.pid = mp.current_process().pid
        self.num_rpc_threads = num_rpc_threads
        self.rpc_timeout = rpc_timeout
        self.num_workers = kwargs.get('num_workers', 0)

        logging.info(f"[{self}] MASTER_ADDR={master_addr}, "
                     f"MASTER_PORT={master_port}")

        if self.num_workers == 0:  # Initialize RPC in main process:
            self.worker_init_fn(0)

    def channel_get(self, out: Any) -> Any:
        pass

    def reset_channel(self, channel=None):
        logging.debug(f'{self} Resetting msg channel')
        while not self.channel.empty():
            self.channel.get_nowait()

        torch.distributed.barrier()

        self.channel = channel or mp.Queue()
        self.dist_sampler.channel = self.channel

    def worker_init_fn(self, worker_id: int):
        try:
            num_sampler_proc = self.num_workers if self.num_workers > 0 else 1
            self.current_ctx_worker = DistContext(
                world_size=self.current_ctx.world_size * num_sampler_proc,
                rank=self.current_ctx.rank * num_sampler_proc + worker_id,
                global_world_size=self.current_ctx.world_size *
                num_sampler_proc,
                global_rank=self.current_ctx.rank * num_sampler_proc +
                worker_id,
                group_name='mp_sampling_worker',
            )

            init_rpc(
                current_ctx=self.current_ctx_worker,
                master_addr=self.master_addr,
                master_port=self.master_port,
                num_rpc_threads=self.num_rpc_threads,
                rpc_timeout=self.rpc_timeout,
            )
            logging.info(
                f"RPC initiated in worker-{worker_id} "
                f"(current_ctx_worker={self.current_ctx_worker.worker_name})")
            self.dist_sampler.init_sampler_instance()
            self.dist_sampler.register_sampler_rpc()
            global_barrier(timeout=10)  # Wait for all workers to initialize.

            atexit.register(shutdown_rpc, self.current_ctx_worker.worker_name)

        except RuntimeError as e:
            raise RuntimeError(f"`{self}.init_fn()` could not initialize the "
                               f"worker loop of the neighbor sampler") from e

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pid={self.pid})'

    def __enter__(self) -> DataLoaderIterator:
        self._prefetch_old = self.prefetch_factor
        self.prefetch_factor = 1
        self._iterator = self._get_iterator()
        return self._iterator

    def __exit__(self, *args) -> None:
        if self.channel:
            self.reset_channel()
        if self._iterator:
            del self._iterator
            torch.distributed.barrier()
            self._iterator = None
            self.prefetch_factor = self._prefetch_old
