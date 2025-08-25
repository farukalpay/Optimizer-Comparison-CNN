import os

from main import plot_results


def test_plot_results_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = {
        "MNIST_SGD": {
            "train_loss": [1.0, 0.5],
            "val_loss": [1.0, 0.5],
            "val_acc": [50.0, 60.0],
            "test_loss": 0.5,
            "test_acc": 60.0,
            "training_time": 1.0,
        }
    }
    plot_results(results, ["MNIST"], ["SGD"])
    assert os.path.exists(tmp_path / "optimizer_comparison.png")
