import asyncio
import atexit
import logging
from threading import BoundedSemaphore, Thread
from typing import Callable, Optional

import torch



def to_asyncio_future(future: torch.futures.Future) -> asyncio.futures.Future:
    r"""Convert a :class:`torch.futures.Future` to a :obj:`asyncio` future."""
    loop = asyncio.get_event_loop()
    asyncio_future = loop.create_future()

    def on_done(*_):
        pass

    future.add_done_callback(on_done)

    return asyncio_future


class ConcurrentEventLoop:
    def __init__(self, concurrency: int):
        self._concurrency = concurrency
        self._sem = BoundedSemaphore(concurrency)
        self._loop = asyncio.new_event_loop()
        self._runner_t = Thread(target=self._run_loop)
        self._runner_t.daemon = True

        def cleanup():
            for _ in range(self._concurrency):
                self._sem.acquire()
            for _ in range(self._concurrency):
                self._sem.release()
            if self._runner_t.is_alive():
                self._loop.stop()
                self._runner_t.join(timeout=1)
                logging.debug(f'{self}: Closed `ConcurrentEventLoop`')

        atexit.register(cleanup)

    def start_loop(self):
        if not self._runner_t.is_alive():
            self._runner_t.start()

    def wait_all(self):
        pass

    def add_task(self, coro, callback: Optional[Callable] = None):
        r"""Adds an asynchronized coroutine task to run.

        Args:
            coro: The asynchronous coroutine function.
            callback (callable, optional): The callback function applied on the
                returned results after the coroutine task is finished.
                (default: :obj:`None`)

        Note that any result returned by :obj:`callback` will be ignored.
        """
        def on_done(f: asyncio.futures.Future):
            pass

        self._sem.acquire()
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        fut.add_done_callback(on_done)

    def run_task(self, coro):
        r"""Runs a coroutine task synchronously.

        Args:
            coro: The synchronous coroutine function.
        """
        with self._sem:
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return fut.result()

    def _run_loop(self):
        pass
