import sys
import traceback
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable, List, Tuple

import pytest
from torch.multiprocessing import Manager, Queue
from typing_extensions import Self


@dataclass
class ProcArgs:
    target: Callable
    args: Tuple[Any, ...]


class MPCaptOutput:
    def __enter__(self) -> Self:
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        sys.stdout = self.stdout
        sys.stderr = self.stderr

        return self

    def __exit__(self, *args: Any) -> None:
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

    @property
    def stdout_str(self) -> str:
        pass

    @property
    def stderr_str(self) -> str:
        pass


def ps_std_capture(
    func: Callable,
    queue: Queue,
    *args: Any,
    **kwargs: Any,
) -> None:
    pass


def assert_run_mproc(
    mp_context: Any,
    pargs: List[ProcArgs],
    full_trace: bool = False,
    timeout: int = 5,
) -> None:
    manager = Manager()
    world_size = len(pargs)
    queues = [manager.Queue() for _ in pargs]
    procs = [
        mp_context.Process(
            target=ps_std_capture,
            args=[p.target, q, world_size] + list(p.args),
        ) for p, q in zip(pargs, queues)
    ]
    results = []

    for p, _ in zip(procs, queues):
        p.start()

    for p, q in zip(procs, queues):
        p.join()
        stdout, stderr = q.get(timeout=timeout)
        results.append((p, stdout, stderr))

    for p, stdout, stderr in results:
        if stdout:
            print(stdout)
        if stderr:  # can be a warning as well => exitcode == 0
            print(stderr)
        if p.exitcode != 0:
            pytest.fail(
                pytrace=full_trace, reason=stderr.splitlines()[-1]
                if stderr else f"exitcode {p.exitcode}")
