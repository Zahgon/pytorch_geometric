import functools
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class ConstantWithWarmupLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        last_epoch: int = -1,
    ):
        lr_lambda = functools.partial(
            self._lr_lambda,
            num_warmup_steps=num_warmup_steps,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        num_warmup_steps: int,
    ) -> float:
        pass


class LinearWithWarmupLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ):
        lr_lambda = functools.partial(
            self._lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        num_warmup_steps: int,
        num_training_steps: int,
    ) -> float:
        pass


class CosineWithWarmupLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        lr_lambda = functools.partial(
            self._lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float,
    ):
        pass


class CosineWithWarmupRestartsLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int = 3,
        last_epoch: int = -1,
    ):
        lr_lambda = functools.partial(
            self._lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int,
    ) -> float:
        pass


class PolynomialWithWarmupLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        lr_end: float = 1e-7,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        lr_init = optimizer.defaults["lr"]
        if not (lr_init > lr_end):
            raise ValueError(f"`lr_end` ({lr_end}) must be smaller than the "
                             f"initial lr ({lr_init})")

        lr_lambda = functools.partial(
            self._lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_init=lr_init,
            lr_end=lr_end,
            power=power,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        num_warmup_steps: int,
        num_training_steps: int,
        lr_init: float,
        lr_end: float,
        power: float,
    ) -> float:
        pass
