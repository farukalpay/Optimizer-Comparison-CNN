import pytest

from optimizer_comparison.optimizers import UnknownOptimizerError, create_optimizer


def test_invalid_optimizer():
    with pytest.raises(UnknownOptimizerError):
        create_optimizer("unknown", [], lr=0.1)
