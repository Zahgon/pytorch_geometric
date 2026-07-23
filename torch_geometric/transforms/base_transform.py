import copy
from abc import ABC, abstractmethod
from typing import Any


class BaseTransform(ABC):
    def __call__(self, data: Any) -> Any:
        return self.forward(copy.copy(data))

    @abstractmethod
    def forward(self, data: Any) -> Any:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
