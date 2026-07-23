import os
import pathlib
import time
from contextlib import ContextDecorator, contextmanager
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import torch
from torch.autograd.profiler import EventList
from torch.profiler import ProfilerActivity, profile

from torch_geometric.profile.utils import (
    byte_to_megabyte,
    get_gpu_memory_from_ipex,
    get_gpu_memory_from_nvidia_smi,
)


@dataclass
class GPUStats:
    time: float
    max_allocated_gpu: float
    max_reserved_gpu: float
    max_active_gpu: float


@dataclass
class CUDAStats(GPUStats):
    nvidia_smi_free_cuda: float
    nvidia_smi_used_cuda: float


@dataclass
class GPUStatsSummary:
    time_mean: float
    time_std: float
    max_allocated_gpu: float
    max_reserved_gpu: float
    max_active_gpu: float


@dataclass
class CUDAStatsSummary(GPUStatsSummary):
    min_nvidia_smi_free_cuda: float
    max_nvidia_smi_used_cuda: float


def profileit(device: str):  # pragma: no cover
    r"""A decorator to facilitate profiling a function, *e.g.*, obtaining
    training runtime and memory statistics of a specific model on a specific
    dataset.
    Returns a :obj:`GPUStats` if :obj:`device` is :obj:`xpu` or extended
    object :obj:`CUDAStats`, if :obj:`device` is :obj:`cuda`.

    Args:
        device (str): Target device for profiling. Options are:
            :obj:`cuda` and obj:`xpu`.

    .. code-block:: python

        @profileit("cuda")
        def train(model, optimizer, x, edge_index, y):
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            return float(loss)

        loss, stats = train(model, x, edge_index, y)
    """
    def decorator(func):
        pass

    return decorator


class timeit(ContextDecorator):
    def __init__(self, log: bool = True, avg_time_divisor: int = 0):
        self.log = log
        self.avg_time_divisor = avg_time_divisor

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t_start = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t_end = time.time()
        self.duration = self.t_end - self.t_start
        if self.avg_time_divisor > 1:
            self.duration = self.duration / self.avg_time_divisor
        if self.log:  # pragma: no cover
            print(f'Time: {self.duration:.8f}s', flush=True)

    def reset(self):
        r"""Prints the duration and resets current timer."""
        if self.t_start is None:
            raise RuntimeError("Timer wasn't started.")
        else:
            self.__exit__()
            self.__enter__()


def get_stats_summary(
    stats_list: Union[List[GPUStats], List[CUDAStats]]
) -> Union[GPUStatsSummary, CUDAStatsSummary]:  # pragma: no cover
    r"""Creates a summary of collected runtime and memory statistics.
    Returns a :obj:`GPUStatsSummary` if list of :obj:`GPUStats` was passed,
    otherwise (list of :obj:`CUDAStats` was passed),
    returns a :obj:`CUDAStatsSummary`.

    Args:
        stats_list (Union[List[GPUStats], List[CUDAStats]]): A list of
            :obj:`GPUStats` or :obj:`CUDAStats` objects, as returned by
            :meth:`~torch_geometric.profile.profileit`.
    """
    kwargs = dict(
        time_mean=float(torch.tensor([s.time for s in stats_list]).mean()),
        time_std=float(torch.tensor([s.time for s in stats_list]).std()),
        max_allocated_gpu=max([s.max_allocated_gpu for s in stats_list]),
        max_reserved_gpu=max([s.max_reserved_gpu for s in stats_list]),
        max_active_gpu=max([s.max_active_gpu for s in stats_list]))

    if all(isinstance(s, CUDAStats) for s in stats_list):
        return CUDAStatsSummary(
            **kwargs,
            min_nvidia_smi_free_cuda=min(
                [s.nvidia_smi_free_cuda for s in stats_list]),
            max_nvidia_smi_used_cuda=max(
                [s.nvidia_smi_used_cuda for s in stats_list]),
        )
    else:
        return GPUStatsSummary(**kwargs)




def read_from_memlab(line_profiler: Any) -> List[float]:  # pragma: no cover
    from pytorch_memlab.line_profiler.line_records import LineRecords


    track_stats = [  # Different statistic can be collected as needed.
        'allocated_bytes.all.peak',
        'reserved_bytes.all.peak',
        'active_bytes.all.peak',
    ]

    records = LineRecords(line_profiler._raw_line_records,
                          line_profiler._code_infos)
    stats = records.display(None, track_stats)._line_records
    return [byte_to_megabyte(x) for x in stats.values.max(axis=0).tolist()]


def trace_handler(p):
    pass


def print_time_total(p):
    pass


def rename_profile_file(*args):
    profile_dir = str(pathlib.Path.cwd()) + '/'
    timeline_file = profile_dir + 'profile'
    for arg in args:
        timeline_file += '-' + arg
    timeline_file += '.json'
    os.rename('timeline.json', timeline_file)


@contextmanager
def torch_profile(export_chrome_trace=True, csv_data=None, write_csv=None):
    use_cuda = torch.cuda.is_available()

    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    if export_chrome_trace:
        p_trace_handler = trace_handler
    else:
        p_trace_handler = print_time_total

    p = profile(activities=activities, on_trace_ready=p_trace_handler)

    with p:
        yield
        p.step()

    if csv_data is not None and write_csv == 'prof':
        if use_cuda:
            profile_sort = 'self_cuda_time_total'
        else:
            profile_sort = 'self_cpu_time_total'
        events = EventList(
            sorted(
                p.key_averages(),
                key=lambda evt: getattr(evt, profile_sort),
                reverse=True,
            ), use_cuda=use_cuda)

        save_profile_data(csv_data, events, use_cuda)


@contextmanager
def xpu_profile(export_chrome_trace=True):
    with torch.autograd.profiler_legacy.profile(use_xpu=True) as profile:
        yield
    print(profile.key_averages().table(sort_by='self_xpu_time_total'))
    if export_chrome_trace:
        profile.export_chrome_trace('timeline.json')


def format_prof_time(time):
    return round(time / 1e6, 3)


def save_profile_data(csv_data, events, use_cuda):
    sum_self_cpu_time_total = sum(
        [event.self_cpu_time_total for event in events])
    sum_cpu_time_total = sum([event.self_cpu_time_total for event in events])
    sum_self_cuda_time_total = sum(
        [event.self_cuda_time_total for event in events]) if use_cuda else 0

    for e in events[:5]:  # Save top 5 most time consuming operations:
        csv_data['NAME'].append(e.key)
        csv_data['SELF CPU %'].append(
            round(e.self_cpu_time_total * 100.0 / sum_self_cpu_time_total, 3))
        csv_data['SELF CPU'].append(format_prof_time(e.self_cpu_time_total))
        csv_data['CPU TOTAL %'].append(
            round(e.cpu_time_total * 100.0 / sum_cpu_time_total, 3))
        csv_data['CPU TOTAL'].append(format_prof_time(e.cpu_time_total))
        csv_data['CPU TIME AVG'].append(format_prof_time(e.cpu_time_total))
        if use_cuda:
            csv_data['SELF CUDA %'].append(e.self_cuda_time_total * 100.0 /
                                           sum_self_cuda_time_total)
            csv_data['SELF CUDA'].append(
                format_prof_time(e.self_cuda_time_total))
            csv_data['CUDA TOTAL'].append(format_prof_time(e.cpu_time_total))
            csv_data['CUDA TIME AVG'].append(format_prof_time(
                e.cpu_time_total))
        csv_data['# OF CALLS'].append(e.count)
