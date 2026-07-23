import functools
from collections import OrderedDict, defaultdict, namedtuple
from typing import Any, List, NamedTuple, Optional, Tuple

import torch
import torch.profiler as torch_profiler

import torch_geometric.typing

Trace = namedtuple('Trace', ['path', 'leaf', 'module'])

Measure = namedtuple('Measure', [
    'self_cpu_total',
    'cpu_total',
    'self_cuda_total',
    'cuda_total',
    'self_cpu_memory',
    'cpu_memory',
    'self_cuda_memory',
    'cuda_memory',
    'occurrences',
])


class Profiler:
    def __init__(
        self,
        model: torch.nn.Module,
        enabled: bool = True,
        use_cuda: bool = False,
        profile_memory: bool = False,
        paths: Optional[List[str]] = None,
    ):
        self._model = model
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.profile_memory = profile_memory
        self.paths = paths

        self.entered = False
        self.exited = False
        self.traces = ()
        self._ids = set()
        self.trace_profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("the profiler can be initialized only once")
        self.entered = True
        self._forwards = {}  # store the original forward functions

        self.traces = tuple(map(self._hook_trace, _walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True

    def get_trace(self):
        pass

    def __repr__(self) -> str:
        return self.get_trace()[0]

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        pass

    def _remove_hook_trace(self, trace):
        pass


def _layer_trace(
        traces: NamedTuple,
        trace_events: Any,
        show_events: bool = True,
        paths: List[str] = None,
        use_cuda: bool = False,
        profile_memory: bool = False,
        dt: Tuple[str, ...] = ('-', '-', '-', ' '),
) -> object:
    pass


def _flatten_tree(t, depth=0):
    pass


def _build_measure_tuple(events: List, occurrences: List) -> NamedTuple:
    pass


def _format_measure_tuple(measure: NamedTuple) -> NamedTuple:
    pass


def _group_by(events, keyfn):
    pass


def _walk_modules(module, name: str = "", path=()):
    if not name:
        name = module.__class__.__name__

    named_children = list(module.named_children())

    path = path + (name, )

    yield Trace(path, len(named_children) == 0, module)

    for name, child_module in named_children:
        yield from _walk_modules(child_module, name=name, path=path)


def format_time(time_us: int) -> str:
    pass


def format_memory(nbytes: int) -> str:
    pass
