# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the DeePoly codebase.

## Overview

DeePoly is a high-order accuracy neural network framework for function approximation and PDE solving. It implements a hybrid "Scoper + Sniper" approach:
- **Scoper**: Neural networks provide global approximation
- **Sniper**: Polynomials refine for high-order accuracy

Key features:
- Hybrid neural network + polynomial methods
- Domain decomposition with segment-wise solving
- Auto-coding capability for PDEs
- GPU acceleration support
- Abstract variable U programming paradigm

## Project Structure

```
DeePoly/
├── src/
│   ├── main_solver.py                 # Main entry point
│   ├── abstract_class/                # Core abstractions
│   │   ├── base_fitter.py            # Base fitting algorithm
│   │   ├── base_net.py               # Neural network base class
│   │   ├── operator_factory.py       # Differential operator generation
│   │   ├── features_generator.py     # Feature generation
│   │   ├── boundary_constraint.py    # Boundary condition handling
│   │   └── config/
│   │       ├── base_data.py          # Data configuration
│   │       └── base_visualize.py     # Visualization base
│   ├── problem_solvers/              # Problem-specific solvers
│   │   ├── func_fitting_solver/      # Function approximation
│   │   ├── linear_pde_solver/        # Linear PDEs
│   │   └── time_pde_solver/          # Time-dependent PDEs
│   │       ├── solver.py             # Main time PDE solver
│   │       ├── core/
│   │       │   ├── net.py            # Neural network implementation
│   │       │   └── fitter.py         # Time-stepping fitter
│   │       ├── time_schemes/         # Time integration schemes
│   │       │   ├── imex_rk_222.py    # IMEX-RK(2,2,2) scheme
│   │       │   └── imex_1st.py       # First-order IMEX
│   │       └── utils/
│   │           └── visualize.py      # Time evolution visualization
│   ├── algebraic_solver/             # Linear algebra backend
│   │   └── linear_solver.py          # GPU-accelerated solver
│   └── meta_coding/                  # Auto-coding utilities
│       └── auto_spotter.py           # Automatic code generation
├── cases/                            # Test cases and examples
│   ├── func_fitting_cases/
│   ├── linear_pde_cases/
│   └── time_pde_cases/
└── CLAUDE.md                         # This file
```

## Quick Start

### Basic Usage
```bash
# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ML

# Run a case
python src/main_solver.py --case_path cases/[category]/[case_name]

# Examples
python src/main_solver.py --case_path cases/func_fitting_cases/case_2d
python src/main_solver.py --case_path cases/linear_pde_cases/poisson_2d
python src/main_solver.py --case_path cases/time_pde_cases/AC_equation
```

### Creating New Cases

1. **Directory Structure**
```
cases/[category]/[case_name]/
├── config.json         # Configuration (REQUIRED)
├── data_generate.py    # Data generation
├── output.py          # Visualization
└── results/           # Output directory
```

2. **Configuration Template**
```json
{
    "problem_type": "time_pde",
    "method": "hybrid", 
    "auto_code": false,
    "eq": {
        "L1": ["diff(u,x,2)"],
        "L2": ["u"],
        "F": ["1"],
        "N": []
    },
    "vars_list": ["u"],
    "spatial_vars": ["x"],
    "Initial_conditions": [
        {
            "var": "u", 
            "value": "sin(pi*x)",
            "points": 100
        }
    ],
    "boundary_conditions": [
        {
            "type": "dirichlet",
            "region": "left",
            "value": "0",
            "points": 1
        }
    ],
    "hidden_dims": [32, 64, 32],
    "epochs_adam": 10000,
    "n_segments": [10],
    "poly_degree": [5], 
    "x_domain": [[-1.0, 1.0]],
    "time_scheme": "IMEX_RK_2_2_2",
    "T": 1.0,
    "dt": 0.01,
    "device": "cuda",
    "linear_device": "cpu"
}
```

## Core Concepts

### Abstract Variable U Programming

The framework uses abstract variable U to represent solution vectors:
- U represents all equation variables abstractly
- Physical quantities (u, v, p, etc.) are components of U
- Operators work on U uniformly without physical interpretation

```python
# Example: U = [u] for scalar equation
# Example: U = [u, v, p] for Navier-Stokes
U_seg[:, j] = features @ coeffs[i, j, :]  # Uniform computation
```

### Domain Decomposition

Solutions are computed segment-wise:
```python
# Each segment has independent polynomial and neural features
for segment_idx in range(n_segments):
    features = self._get_features(segment_idx, model)
    U_segment = solve_segment(features, coeffs)
```

### Operator Factory Pattern

Automatic generation of differential operators:
```python
operator_factory = create_operator_factory(
    all_derivatives=config.all_derivatives,
    constants=config.constants,
    optimized=True
)
L1, L2, N, F = operator_factory.create_all_operators(operator_terms)
```

## Time PDE Solver

### IMEX Time Stepping

The solver uses IMEX (Implicit-Explicit) schemes for time integration:
- **L1**: Linear stiff terms (implicit)
- **L2**: Linear multiplicative terms
- **F**: Source terms
- **N**: Nonlinear terms (explicit)

Time evolution equation:
```
∂U/∂t = L1(U) + L2(U)*F(U) + N(U)
```

### Key Features

1. **Reference Solution Comparison**
   - Load MATLAB `.mat` reference data
   - Real-time error monitoring
   - Comprehensive error analysis

2. **Spotter Skip Optimization**
   - Skip neural network training at specified intervals
   - Reuse previous network parameters
   - Accelerate time stepping

3. **Boundary Conditions**
   - Dirichlet, Neumann, Robin, Periodic
   - Modular implementation in `net.py`
   - Cross-segment constraints for periodic BC

### Configuration for Time PDEs
```json
{
    "problem_type": "time_pde",
    "method": "hybrid",
    "auto_code": false,
    "eq": {
        "L1": ["0.0001*diff(u,x,2)"],
        "L2": ["u"],
        "F": ["5-5*u**2"],
        "N": []
    },
    "vars_list": ["u"],
    "spatial_vars": ["x"],
    "Initial_conditions": [
        {
            "var": "u",
            "value": "x**2*cos(pi*x)",
            "points": 100
        }
    ],
    "boundary_conditions": [
        {
            "type": "periodic",
            "pairs": ["left", "right"],
            "points": 1
        }
    ],
    "time_scheme": "IMEX_RK_2_2_2",
    "T": 0.5,
    "dt": 0.1,
    "spotter_skip": 1,
    "reference_solution": "reference_data/allen_cahn_highres.mat",
    "realtime_visualization": "True",
    "animation_skip": 1
}
```

## Performance Guidelines

### Memory Management
- **Tensor Dimensions**:
  - `features`: `(n_points, dgN)` per segment
  - `coeffs`: `(ns, n_eqs, dgN)`
  - `U_seg`: `(n_points, n_eqs)`
- **Avoid dimension branching**: Use consistent tensor operations
- **Print dimensions for debugging**: Don't use excessive type checking

### Device Configuration
- **Neural Networks**: `"device": "cuda"` (GPU preferred)
- **Linear Algebra**: `"linear_device": "cpu"` (stability)
- **Memory Limits**: Monitor GPU usage, adjust batch sizes

### Optimization Tips
1. Start with small `epochs_adam` (1000-5000) for testing
2. Use `spotter_skip` for time PDEs to reduce training overhead
3. Adjust `n_segments` based on solution complexity
4. Keep `poly_degree` moderate (5-7) for stability

## Common Issues and Solutions

### Training Issues
| Issue | Solution |
|-------|----------|
| Loss not converging | Reduce learning rate, check boundary conditions |
| NaN values | Use CPU for linear algebra, reduce learning rate |
| CUDA out of memory | Reduce `points_domain`, use smaller `hidden_dims` |
| Slow convergence | Increase polynomial degree, add more segments |

### Configuration Errors
- Ensure `len(x_domain) == len(spatial_vars)`
- Check `problem_type` is valid: `func_fitting`, `linear_pde`, or `time_pde`
- Verify all required fields are present in `config.json`

### Debugging Commands
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Validate configuration
python -c "import json; json.load(open('config.json'))"

# Test import
python -c "from src.abstract_class.base_fitter import BaseDeepPolyFitter"
```

## PDE Configuration Format

### New Structured Format

The framework now uses a structured approach for defining PDEs with separate operator categories:

```json
{
    "eq": {
        "L1": ["0.0001*diff(u,x,2)"],           // Linear stiff terms (implicit)
        "L2": ["u"],                            // Linear multiplicative terms  
        "F": ["5-5*u**2"],                     // Source/forcing terms
        "N": []                                // Nonlinear terms (explicit)
    },
    "vars_list": ["u"],                        // Solution variables
    "Initial_conditions": [
        {
            "var": "u",
            "value": "x**2*cos(pi*x)",         // Initial condition expression
            "points": 100
        }
    ],
    "boundary_conditions": [
        {
            "type": "periodic",                 // BC type: periodic, dirichlet, neumann
            "pairs": ["left", "right"],        // Paired boundary regions
            "points": 1                        // Number of boundary points
        }
    ]
}
```

### Auto-Coding for Linear PDEs

For linear PDEs, the framework can automatically generate solving code:

1. Set `"auto_code": true` in config.json
2. Define PDE using the structured format above
3. Run once to generate operator code
4. Set `"auto_code": false` for subsequent runs

### Mathematical Notation

- **Derivatives**: `diff(u,x,2)` for ∂²u/∂x²
- **Variables**: Use variable names from `vars_list`
- **Constants**: Numeric values and `pi`, `e`
- **Functions**: `sin`, `cos`, `exp`, etc.
- **Powers**: Use `**` notation (e.g., `u**2`)

## Code Style Guidelines

1. **Clean and Efficient**: Prioritize readability and performance
2. **Minimal Branching**: Avoid excessive conditional checks
3. **Consistent Tensors**: Use uniform tensor operations
4. **English Comments**: All documentation in English
5. **Dimension Verification**: Print shapes during debugging, not runtime checks

## Version Information

**Current Version**: v0.2
- High-accuracy arbitrary-dimensional PDE solving
- IMEX-RK time integration schemes
- Comprehensive boundary condition support
- Reference solution comparison
- Full English documentation

## Important Notes

- Always verify tensor dimensions by printing during development
- Avoid matrix/array dimension type branching in production code
- Use abstract variable U consistently throughout the codebase
- Maintain separation between physical interpretation and numerical computation
- 保持全部代码文件英文书写,不要出现中文字符