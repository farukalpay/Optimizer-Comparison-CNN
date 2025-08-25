# Optimizer Comparison for CNN Training

This repository contains a PyTorch implementation for comparing three optimization algorithms (SGD with Momentum, Adam, and a custom Perturbed Saddle-escape Descent) on MNIST and CIFAR-10 datasets.

## Features

- Implementation of custom Perturbed Saddle-escape Descent (PSD) optimizer
- Training and evaluation of CNN on MNIST and CIFAR-10 datasets
- Comprehensive comparison of optimization algorithms
- Visualization of training dynamics and performance metrics

## Requirements

- Python 3.6+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy

## Installation

```bash
git clone https://github.com/your-username/Optimizer-Comparison-CNN.git
cd Optimizer-Comparison-CNN
pip install -r requirements.txt
```

Usage

Run the main experiment script:

```bash
python optimizer_comparison.py
```

This will:

1. Train a CNN on both MNIST and CIFAR-10 datasets
2. Use three different optimizers (SGD, Adam, PSD)
3. Generate comparison plots of training metrics
4. Save results to optimizer_comparison.png

Results

The script generates comparative plots showing:

· Training loss over epochs
· Validation loss over epochs
· Validation accuracy over epochs

Custom PSD Optimizer

The implementation includes a custom Perturbed Saddle-escape Descent optimizer based on:

· "How to Escape Saddle Points Efficiently" by Alpay & Alakkad 2025
· GitHub: https://github.com/farukalpay/PSD

License

MIT License - see LICENSE file for details

```

## Key Files

- `optimizer_comparison.py` - Main experiment script
- `requirements.txt` - Python dependencies
- `README.md` - This file

The repository provides a complete implementation for comparing optimization algorithms with a focus on reproducibility and clear visualization of results.
