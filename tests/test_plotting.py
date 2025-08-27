from pathlib import Path

import pandas as pd

from optimizer_comparison.plotting import plot_metrics


def test_plot_metrics(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {"epoch": [0, 1], "val_loss": [1.0, 0.5], "val_acc": [0.1, 0.2]}
    )
    out = tmp_path / "plot.png"
    plot_metrics({"sgd": data}, out)
    assert out.exists()
