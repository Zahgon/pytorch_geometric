import functools
import inspect
import warnings
from typing import Any, Callable, Optional


def deprecated(
    details: Optional[str] = None,
    func_name: Optional[str] = None,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        pass

    return decorator
