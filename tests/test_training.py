import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from hypothesis import given, strategies as st
import pytest

from main import SimpleCNN, train_one_epoch, evaluate


def _constant_dataset(size: int):
    inputs = torch.randn(size, 1, 28, 28)
    targets = torch.zeros(size, dtype=torch.long)
    return DataLoader(TensorDataset(inputs, targets), batch_size=max(1, size // 2))


def test_train_one_epoch_updates_parameters():
    model = SimpleCNN(num_classes=10, input_channels=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = _constant_dataset(8)
    params_before = [p.clone() for p in model.parameters()]
    loss = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    params_after = list(model.parameters())
    assert loss >= 0
    assert any(not torch.equal(a, b) for a, b in zip(params_before, params_after))


class ConstantModel(nn.Module):
    def __init__(self, label: int = 0):
        super().__init__()
        self.label = label

    def forward(self, x):
        out = torch.zeros(x.size(0), 10)
        out[:, self.label] = 1.0
        return out


@given(st.integers(min_value=1, max_value=5))
def test_evaluate_perfect_accuracy(batch):
    model = ConstantModel(label=0)
    criterion = nn.CrossEntropyLoss()
    loader = _constant_dataset(batch)
    loss, acc = evaluate(model, loader, criterion, torch.device("cpu"))
    assert acc == 100.0


def test_evaluate_empty_dataloader_raises():
    model = ConstantModel(label=0)
    criterion = nn.CrossEntropyLoss()
    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, 1, 28, 28), torch.empty(0, dtype=torch.long)),
        batch_size=1,
    )
    with pytest.raises(ValueError):
        evaluate(model, empty_loader, criterion, torch.device("cpu"))
