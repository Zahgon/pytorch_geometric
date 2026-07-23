from typing import Any

__debug_flag__ = {'enabled': False}


def is_debug_enabled() -> bool:
    r"""Returns :obj:`True` if the debug mode is enabled."""
    return __debug_flag__['enabled']


def set_debug_enabled(mode: bool) -> None:
    __debug_flag__['enabled'] = mode


class debug:
    def __init__(self) -> None:
        self.prev = is_debug_enabled()

    def __enter__(self) -> None:
        set_debug_enabled(True)

    def __exit__(self, *args: Any) -> None:
        set_debug_enabled(self.prev)


class set_debug:
    def __init__(self, mode: bool) -> None:
        self.prev = is_debug_enabled()
        set_debug_enabled(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> None:
        set_debug_enabled(self.prev)
