"""Utilities for comparing optimizers on simple CNNs."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - during local development package metadata may be missing
    __version__ = version("optimizer-comparison-cnn")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
