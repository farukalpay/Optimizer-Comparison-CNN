"""Command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
import yaml

from .plotting import plot_from_summary
from .train import RunConfig
from .train import run as run_experiment

app = typer.Typer(help="Compare optimizers on simple CNN benchmarks")


def load_config(path: Path) -> RunConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return RunConfig(**data)


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, help="Path to YAML config"),
    dataset: str = typer.Option("mnist"),
    optimizer: List[str] = typer.Option(
        ["sgd", "adam", "psd"], help="Optimizers to run"
    ),
    epochs: int = typer.Option(1),
    batch_size: int = typer.Option(128),
    lr: float = typer.Option(0.01),
    out: Path = typer.Option(Path("runs")),
    seed: int = typer.Option(0),
    device: Optional[str] = typer.Option(None),
    fake_data: bool = typer.Option(False, help="Use FakeData dataset"),
) -> None:
    """Run training for one dataset and multiple optimizers."""
    if config:
        cfg = load_config(config)
    else:
        cfg = RunConfig(
            dataset=dataset,
            optimizers=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            out_dir=str(out),
            seed=seed,
            device=device,
            fake_data=fake_data,
        )
    run_experiment(cfg)


@app.command()
def plot(summary: Path) -> None:
    """Regenerate comparison plot from a summary.csv file."""
    out_path = plot_from_summary(summary)
    typer.echo(f"saved plot to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
