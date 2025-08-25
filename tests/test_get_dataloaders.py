import pytest

from main import get_dataloaders


def test_get_dataloaders_invalid_dataset():
    with pytest.raises(ValueError):
        get_dataloaders("INVALID", 32)
