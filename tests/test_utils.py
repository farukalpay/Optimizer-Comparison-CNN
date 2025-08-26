import numpy as np
import torch
import pytest

from utils import set_seed, get_device
from main import create_optimizer, SimpleCNN


def test_set_seed_reproducible():
    set_seed(123)
    a1 = torch.rand(2)
    b1 = np.random.rand(2)
    set_seed(123)
    a2 = torch.rand(2)
    b2 = np.random.rand(2)
    assert torch.equal(a1, a2)
    assert np.allclose(b1, b2)


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_create_optimizer_invalid_name():
    model = SimpleCNN(num_classes=10, input_channels=1)
    with pytest.raises(ValueError):
        create_optimizer("INVALID", model.parameters())
