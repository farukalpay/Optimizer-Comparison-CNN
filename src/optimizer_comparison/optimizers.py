"""Optimizer factory."""

from __future__ import annotations

from typing import Iterable

import torch.optim as optim
from psd_optimizer import PSDOptimizer


class UnknownOptimizerError(ValueError):
    """Raised when an optimizer name is not recognised."""


def create_optimizer(
    name: str, params: Iterable, lr: float, momentum: float = 0.9, **kwargs
):
    """Create an optimizer from its string identifier."""
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum)
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "psd":
        return PSDOptimizer(
            params,
            lr=lr,
            epsilon=kwargs.get("epsilon", 1e-3),
            r=kwargs.get("r", 1e-4),
            T=kwargs.get("T", 10),
        )
    raise UnknownOptimizerError(name)
