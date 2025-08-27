"""Dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(
    dataset: str,
    batch_size: int,
    data_dir: str = "./data",
    num_workers: int = 0,
    fake_data: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test dataloaders for the requested dataset.

    When ``fake_data`` is True a small :class:`torchvision.datasets.FakeData`
    dataset is returned.  This is primarily used for tests to avoid downloads.
    """
    if fake_data:
        dataset_obj = datasets.FakeData(
            size=256,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transforms.ToTensor(),
        )
        train_set, val_set, test_set = random_split(dataset_obj, [128, 64, 64])
    else:
        root = Path(data_dir)
        if dataset.lower() == "mnist":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            full_train = datasets.MNIST(
                root=root, train=True, download=True, transform=transform
            )
            test_set = datasets.MNIST(
                root=root, train=False, download=True, transform=transform
            )
            train_set, val_set = random_split(full_train, [50_000, 10_000])
        elif dataset.lower() == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            full_train = datasets.CIFAR10(
                root=root, train=True, download=True, transform=transform
            )
            test_set = datasets.CIFAR10(
                root=root, train=False, download=True, transform=transform
            )
            train_set, val_set = random_split(full_train, [45_000, 5_000])
        else:  # pragma: no cover - handled by CLI validation
            raise ValueError(f"unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
