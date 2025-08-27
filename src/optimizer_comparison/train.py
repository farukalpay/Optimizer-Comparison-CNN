"""Training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .data import get_dataloaders
from .models import SimpleCNN
from .optimizers import create_optimizer


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:  # pragma: no cover - optional
        torch.use_deterministic_algorithms(False)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return available device prioritising CUDA then MPS then CPU."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module, loader: Iterable, optimizer, criterion, device: torch.device
) -> float:
    model.train()
    running_loss = 0.0
    for batch in loader:
        inputs, targets = (t.to(device) for t in batch)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module, loader: Iterable, criterion, device: torch.device
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
    return {
        "loss": loss_sum / len(loader.dataset),
        "acc": correct / len(loader.dataset),
    }


@dataclass
class RunConfig:
    dataset: str = "mnist"
    optimizers: List[str] = None  # type: ignore[assignment]
    epochs: int = 1
    batch_size: int = 128
    lr: float = 0.01
    out_dir: str = "runs"
    seed: int = 0
    device: Optional[str] = None
    fake_data: bool = False

    def __post_init__(self) -> None:
        if self.optimizers is None:
            self.optimizers = ["sgd", "adam", "psd"]


def run(config: RunConfig) -> Path:
    """Execute training for the requested optimizers and dataset."""
    set_seed(config.seed)
    device = get_device(config.device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(config.out_dir) / timestamp / config.dataset
    summary: List[Dict[str, float]] = []
    root.mkdir(parents=True, exist_ok=True)

    for opt_name in config.optimizers:
        train_loader, val_loader, _ = get_dataloaders(
            config.dataset, config.batch_size, fake_data=config.fake_data
        )
        in_channels = (
            3 if config.fake_data else (1 if config.dataset.lower() == "mnist" else 3)
        )
        model = SimpleCNN(in_channels=in_channels).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(opt_name, model.parameters(), lr=config.lr)
        metrics: List[Dict[str, float]] = []
        for epoch in tqdm(range(config.epochs), desc=f"{opt_name}"):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_stats = evaluate(model, val_loader, criterion, device)
            metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_stats["loss"],
                    "val_acc": val_stats["acc"],
                }
            )
        opt_dir = root / opt_name
        opt_dir.mkdir(parents=True, exist_ok=True)
        import csv

        with open(opt_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"]
            )
            writer.writeheader()
            writer.writerows(metrics)
        summary.append({"optimizer": opt_name, **metrics[-1]})

    with open(root.parent / "summary.csv", "w", newline="") as f:
        import csv

        writer = csv.DictWriter(
            f, fieldnames=["optimizer", "epoch", "train_loss", "val_loss", "val_acc"]
        )
        writer.writeheader()
        writer.writerows(summary)
    return root.parent
