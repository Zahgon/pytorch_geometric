import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from torch_geometric.utils import *  # noqa

__experimental_flag__: Dict[str, bool] = {
    'disable_dynamic_shapes': False,
}

Options = Optional[Union[str, List[str]]]


def get_options(options: Options) -> List[str]:
    if options is None:
        options = list(__experimental_flag__.keys())
    if isinstance(options, str):
        options = [options]
    return options


def is_experimental_mode_enabled(options: Options = None) -> bool:
    r"""Returns :obj:`True` if the experimental mode is enabled. See
    :class:`torch_geometric.experimental_mode` for a list of (optional)
    options.
    """
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return False
    options = get_options(options)
    return all([__experimental_flag__[option] for option in options])


def set_experimental_mode_enabled(mode: bool, options: Options = None) -> None:
    for option in get_options(options):
        __experimental_flag__[option] = mode


class experimental_mode:
    def __init__(self, options: Options = None) -> None:
        self.options = get_options(options)
        self.previous_state = {
            option: __experimental_flag__[option]
            for option in self.options
        }

    def __enter__(self) -> None:
        set_experimental_mode_enabled(True, self.options)

    def __exit__(self, *args: Any) -> None:
        for option, value in self.previous_state.items():
            __experimental_flag__[option] = value


class set_experimental_mode:
    def __init__(self, mode: bool, options: Options = None) -> None:
        self.options = get_options(options)
        self.previous_state = {
            option: __experimental_flag__[option]
            for option in self.options
        }
        set_experimental_mode_enabled(mode, self.options)

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> None:
        for option, value in self.previous_state.items():
            __experimental_flag__[option] = value


def disable_dynamic_shapes(required_args: List[str]) -> Callable:
    r"""A decorator that disables the usage of dynamic shapes for the given
    arguments, i.e., it will raise an error in case :obj:`required_args` are
    not passed and needs to be automatically inferred.
    """
    def decorator(func: Callable) -> Callable:
        pass

    return decorator
