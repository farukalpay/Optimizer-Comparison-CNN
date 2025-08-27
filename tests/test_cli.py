from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from optimizer_comparison.cli import app


def test_run_fake(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--fake-data",
            "--optimizer",
            "sgd",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    summary_files = list(tmp_path.glob("*/summary.csv"))
    assert summary_files, "summary.csv not created"


def test_plot(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "run",
            "--fake-data",
            "--optimizer",
            "sgd",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--out",
            str(tmp_path),
        ],
    )
    summary = next(tmp_path.glob("*/summary.csv"))
    result = runner.invoke(app, ["plot", str(summary)])
    assert result.exit_code == 0
    assert (summary.parent / "optimizer_comparison.png").exists()
