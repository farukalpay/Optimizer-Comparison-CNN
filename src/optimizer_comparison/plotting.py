"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")


def plot_metrics(metrics: Dict[str, pd.DataFrame], out_file: Path) -> None:
    """Plot loss and accuracy curves for each optimizer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, df in metrics.items():
        ax1.plot(df["epoch"], df["val_loss"], label=name)
        ax2.plot(df["epoch"], df["val_acc"], label=name)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Acc")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def plot_from_summary(summary_file: Path) -> Path:
    """Rebuild comparison plot from a summary.csv file."""
    timestamp_dir = summary_file.parent
    dataset_dirs = [p for p in timestamp_dir.iterdir() if p.is_dir()]
    if not dataset_dirs:  # pragma: no cover - defensive
        raise FileNotFoundError("no dataset directory found")
    dataset_dir = dataset_dirs[0]
    metrics: Dict[str, pd.DataFrame] = {}
    for opt_dir in dataset_dir.iterdir():
        if opt_dir.is_dir():
            metrics[opt_dir.name] = pd.read_csv(opt_dir / "metrics.csv")
    out_file = timestamp_dir / "optimizer_comparison.png"
    plot_metrics(metrics, out_file)
    return out_file
