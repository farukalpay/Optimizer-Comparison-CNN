# Optimizer Comparison for CNN Training

This repository provides a minimal yet extensible baseline for comparing
three optimization algorithms – SGD with momentum, Adam and the
[Perturbed Saddle Descent (PSD)](https://pypi.org/project/psd-optimizer/)
optimizer – on small convolutional neural networks using PyTorch.

The PSD algorithm is described in the paper
["How to Escape Saddle Points Efficiently"](https://arxiv.org/abs/2508.16540).

## Quickstart

```bash
pip install -e .[dev]
optimizer-cmp run --fake-data --epochs 1
```

This trains a tiny CNN on a synthetic dataset using all three optimizers
and stores results under `runs/<timestamp>`.

## CLI Usage

- `optimizer-cmp run` – train one dataset with one or more optimizers.
- `optimizer-cmp plot <summary.csv>` – regenerate the comparison plot
  from a previous run.

For full options see `optimizer-cmp run --help`.

## License

MIT
