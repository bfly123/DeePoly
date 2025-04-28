# DeePoly - High-Order Accuracy Neural Network Framework for Function Approximation and PDE Solving

![](/Doc/logo_deepoly.png)

## Introduction

DeePoly is a novel general-purpose platform for function approximation and equation solving algorithms. Core algorithm: [arxiv]

## Key Features

- **Mesh-Free**: Sampling points can be randomly generated with no logical relationships, suitable for complex geometries.
- **High Accuracy**: Achieves high-order convergence.
- **Format-Free**: Handles derivative relationships using automatic differentiation.
- **Efficient**: Computational efficiency comparable to traditional finite difference methods.
- **GPU Accelerated**: Supports CPU parallelism and GPU acceleration.
- **Applicable to Complex and Discontinuous Problems**: Accurately approximates discontinuous and high-gradient functions.
- **Suitable for Inverse Problems**: Solves inverse problems with higher accuracy than PINNs.

## Version Information

Current version: v0.1 (Beta). The `cases` directory includes only function approximation test cases, but the core algorithm in `src` already contains all derivative computation code. PDE solving examples have undergone preliminary independent testing, and an integrated version will be released soon.

Upcoming v0.2:
- High-accuracy solving for arbitrary-dimensional linear PDEs.
- English-commented version.

## Usage

Develop your own problems in the `cases` directory, including data generation, output, and configuration files.

## Installation Requirements

- Python
- CUDA
- torch

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/deepoly.git

# Run example
python src/main_solver.py --case_path [your_case_path]
