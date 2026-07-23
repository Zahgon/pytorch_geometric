import importlib
import inspect
import re
from typing import Any, List, Optional


def paper_link(cls: str) -> Optional[str]:
    pass


def get_stats_table(cls: str) -> str:
    pass


def has_stats(cls: str) -> bool:
    pass


def get_type(cls: str) -> str:
    pass


def get_stat(cls: str, name: str, child: Optional[str] = None,
             default: Any = None) -> str:
    pass


def get_children(cls: str) -> List[str]:
    pass
