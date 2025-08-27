from __future__ import annotations

import pandas as pd

from optimizer_comparison.train import RunConfig, run


def test_deterministic_metrics(tmp_path):
    cfg = RunConfig(
        epochs=1,
        batch_size=16,
        fake_data=True,
        out_dir=str(tmp_path),
        seed=123,
        optimizers=["sgd"],
    )
    run(cfg)
    summary1 = next(tmp_path.glob("*/summary.csv"))
    df1 = pd.read_csv(summary1)
    # run again
    run(cfg)
    summary2 = sorted(tmp_path.glob("*/summary.csv"))[-1]
    df2 = pd.read_csv(summary2)
    assert df1.equals(df2)
