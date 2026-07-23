from functools import wraps
from typing import Optional

import torch

CUDA_PROFILE_STARTED = False


def begin_cuda_profile():
    global CUDA_PROFILE_STARTED
    prev_state = CUDA_PROFILE_STARTED
    if prev_state is False:
        CUDA_PROFILE_STARTED = True
        torch.cuda.cudart().cudaProfilerStart()
    return prev_state


def end_cuda_profile(prev_state: bool):
    global CUDA_PROFILE_STARTED
    CUDA_PROFILE_STARTED = prev_state
    if prev_state is False:
        torch.cuda.cudart().cudaProfilerStop()


def nvtxit(name: Optional[str] = None, n_warmups: int = 0,
           n_iters: Optional[int] = None):
    """Enables NVTX profiling for a function.

    Args:
        name (Optional[str], optional): Name to give the reference frame for
            the function being wrapped. Defaults to the name of the
            function in code.
        n_warmups (int, optional): Number of iters to call that function
            before starting. Defaults to 0.
        n_iters (Optional[int], optional): Number of iters of that function to
            record. Defaults to all of them.
    """
    def nvtx(func):
        pass

    return nvtx
