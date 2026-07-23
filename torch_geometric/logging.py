import sys
from typing import Any

_wandb_initialized: bool = False


def init_wandb(name: str, **kwargs: Any) -> None:
    pass


def log(**kwargs: Any) -> None:
    def _map(value: Any) -> str:
        if isinstance(value, int) and not isinstance(value, bool):
            return f'{value:03d}'
        if isinstance(value, float):
            return f'{value:.4f}'
        return value

    print(', '.join(f'{key}: {_map(value)}' for key, value in kwargs.items()))

    if _wandb_initialized:
        import wandb
        wandb.log(kwargs)
