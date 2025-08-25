import torch
import torchvision
import pytest

from main import get_dataloaders


def _fake_dataset_factory(channels: int, height: int, width: int, train_size: int, test_size: int):
    class _FakeDataset(torchvision.datasets.FakeData):
        def __init__(self, root=None, train: bool = True, download: bool = False, transform=None):
            size = train_size if train else test_size
            super().__init__(
                size=size,
                image_size=(channels, height, width),
                num_classes=10,
                transform=transform,
            )

    return _FakeDataset


@pytest.fixture(autouse=True)
def patch_datasets_and_loader(monkeypatch):
    monkeypatch.setattr(
        torchvision.datasets,
        "MNIST",
        _fake_dataset_factory(1, 28, 28, 60000, 10000),
    )
    monkeypatch.setattr(
        torchvision.datasets,
        "CIFAR10",
        _fake_dataset_factory(3, 32, 32, 50000, 10000),
    )

    original_loader = torch.utils.data.DataLoader

    def _loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", _loader)


def _assert_batch(
    loader: torch.utils.data.DataLoader,
    channels: int,
    height: int,
    width: int,
    batch_size: int,
):
    images, labels = next(iter(loader))
    assert images.shape == (batch_size, channels, height, width)
    assert images.dtype == torch.float32
    assert labels.shape == (batch_size,)
    assert labels.dtype == torch.int64


def test_get_dataloaders_mnist():
    train, val, test = get_dataloaders("MNIST", batch_size=4)
    _assert_batch(train, 1, 28, 28, 4)
    _assert_batch(val, 1, 28, 28, 4)
    _assert_batch(test, 1, 28, 28, 4)


def test_get_dataloaders_cifar10():
    train, val, _ = get_dataloaders("CIFAR10", batch_size=4)
    _assert_batch(train, 3, 32, 32, 4)
    _assert_batch(val, 3, 32, 32, 4)


def test_get_dataloaders_invalid_dataset():
    with pytest.raises(ValueError):
        get_dataloaders("INVALID", 32)

