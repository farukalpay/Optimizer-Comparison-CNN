# Optimizer Comparison for CNN Training

This repository contains a PyTorch implementation for comparing three optimization algorithms (SGD with Momentum, Adam, and the Perturbed Saddle Descent optimizer from the [psd-optimizer](https://pypi.org/project/psd-optimizer/) library) on MNIST and CIFAR-10 datasets.

## Features

- Integration of the [psd-optimizer](https://pypi.org/project/psd-optimizer/) library
- Training and evaluation of CNN on MNIST and CIFAR-10 datasets
- Comprehensive comparison of optimization algorithms
- Visualization of training dynamics and performance metrics
- macOS-friendly defaults including support for Apple's Metal (MPS) backend and single-process data loading
- Helper functions for reproducible experiments and device selection

## Requirements

- Python 3.6+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy
- psd-optimizer

## Installation

```bash
git clone https://github.com/your-username/Optimizer-Comparison-CNN.git
cd Optimizer-Comparison-CNN
pip install -r requirements.txt
```

## Usage

Run the main experiment script:

```bash
python main.py
```

This will:

1. Train a CNN on both MNIST and CIFAR-10 datasets
2. Use three different optimizers (SGD, Adam, PSD)
3. Generate comparison plots of training metrics
4. Save results to optimizer_comparison.png

## Utility Functions

The project includes small utility helpers to make experiments more reliable:

- `set_seed(seed)` – configure random seeds across libraries for reproducibility.
- `get_device()` – automatically choose between CUDA, Apple's MPS, or CPU.
- `create_optimizer(name, params)` – build a configured optimizer and validate its name.

## Results

The script generates comparative plots showing:

· Training loss over epochs
· Validation loss over epochs
· Validation accuracy over epochs

## PSD Optimizer

This project uses the `psd-optimizer` package, which implements the Perturbed Saddle Descent (PSD) algorithm. The algorithm is described in:

- Alpay and Alakkad, 2025, "Escaping Saddle Points via Curvature-Calibrated Perturbations: A Complete Analysis with Explicit Constants and Empirical Validation" ([arXiv:2508.16540](https://arxiv.org/abs/2508.16540))
- GitHub: https://github.com/farukalpay/PSD

## License

MIT License - see LICENSE file for details

## Key Files

- `main.py` - Main experiment script
- `requirements.txt` - Python dependencies
- `README.md` - This file

The repository provides a complete implementation for comparing optimization algorithms with a focus on reproducibility and clear visualization of results.
