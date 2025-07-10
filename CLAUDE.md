# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DeePoly is a high-order accuracy neural network framework for function approximation and PDE solving. It implements a hybrid approach combining traditional polynomials with neural networks to achieve both high accuracy and computational efficiency. The core innovation is the "Scoper + Sniper" two-phase approach: neural networks provide global approximation (Scoper), followed by polynomial refinement for high-order accuracy (Sniper).

## Architecture

### Entry Point
- Main entry: `src/main_solver.py`
- Usage: `python src/main_solver.py --case_path cases/[category]/[case_name]`

### Core Components
- **BaseDeepPolyFitter** (`src/abstract_class/base_fitter.py`): Core fitting algorithm with hierarchical pre-compilation
- **Problem Solvers** (`src/problem_solvers/`): Specialized solvers for different problem types
- **LinearSolver** (`src/algebraic_solver/linear_solver.py`): GPU-accelerated linear algebra backend
- **OperatorFactory** (`src/abstract_class/operator_factory.py`): Automatic differential operator generation
- **FeatureGenerator** (`src/abstract_class/features_generator.py`): Polynomial and neural network feature generation

### Problem Types
- **func_fitting**: Function approximation problems
- **linear_pde**: Linear PDEs with auto-coding capability
- **time_pde**: Time-dependent PDEs

## Development Commands

### Running Cases
```bash
# Run function fitting example
python src/main_solver.py --case_path cases/func_fitting_cases/case_2d

# Run PDE solving example
python src/main_solver.py --case_path cases/linear_pde_cases/poisson_2d_sinpixsinpiy

# For new linear PDE cases: Set auto_code=true in config.json on first run, then rerun with auto_code=false
```

### Testing and Validation
```bash
# Quick test with minimal epochs
python src/main_solver.py --case_path cases/func_fitting_cases/test_sin

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Validate configuration
python -c "import json; print(json.load(open('cases/your_case/config.json')))"
```

## Configuration Management

### Essential Config Fields
Every `config.json` must contain:
- `problem_type`: "func_fitting", "linear_pde", or "time_pde"
- `method`: "hybrid" (neural + polynomial), "poly" (polynomial only), or "dnn" (neural only)
- `spatial_vars`: Array of spatial variables (e.g., ["x", "y"])
- `hidden_dims`: Neural network architecture (e.g., [32, 64, 32])
- `epochs_adam`: Training epochs (start with 10000-30000)
- `x_domain`: Domain boundaries for each dimension
- `device`: "cuda" or "cpu" for neural networks
- `linear_device`: "cpu" recommended for linear algebra stability

### Problem-Specific Fields
- Function fitting: `points_domain`, `poly_degree`, `n_segments`
- Linear PDE: `auto_code`, `pde_equation`, `boundary_conditions`
- Time PDE: `time_domain`, `dt`, `time_scheme`

## Code Patterns

### Configuration Loading
```python
@dataclass
class BaseConfig:
    problem_type: str
    method: str
    spatial_vars: List[str]
    x_domain: List[List[float]]
    device: str
    
    def __post_init__(self):
        self.n_dim = len(self.spatial_vars)
        self.validate()
```

### Operator Factory Pattern
```python
operator_factory = create_operator_factory(
    all_derivatives=config.all_derivatives,
    constants=config.constants,
    optimized=True
)
operators = operator_factory.create_all_operators(operator_terms)
```

### Feature Generation
```python
# Get features for segment (returns [polynomial_features, neural_features])
features = self._get_features(segment_idx, model)
```

## Adding New Cases

### Case Structure
```
cases/[category]/[case_name]/
├── config.json         # Configuration (REQUIRED)
├── data_generate.py    # Data generation script
├── output.py          # Output and visualization
├── run_experiments.py # Experiment runner
└── results/           # Generated results
```

### Required Functions
- `data_generate.py`: Must define `target_function(x)` and `generate_data()`
- `output.py`: Must define `output_results(solver, config)`

### Workflow
1. Create directory structure
2. Copy template files from similar case
3. Modify config.json with problem parameters
4. Update data_generate.py with target function/PDE
5. For linear PDEs: Set `auto_code: true` on first run

## Common Issues and Solutions

### Configuration Errors
- **Missing fields**: Ensure all required fields are present
- **Dimension mismatch**: `len(x_domain) == len(spatial_vars)`
- **Invalid problem_type**: Must be one of the three supported types

### Training Issues
- **Loss not converging**: Check boundary conditions, reduce learning rate, try "poly" method
- **NaN values**: Reduce learning rate, use `"linear_device": "cpu"`
- **CUDA out of memory**: Reduce `points_domain` or `hidden_dims`, use CPU

### Performance Optimization
- Use `"device": "cuda"` for neural networks when available
- Use `"linear_device": "cpu"` for numerical stability
- Start with small epochs for testing
- Use appropriate `n_segments` for domain complexity

## Mathematical Equation Format

### Supported PDE Operators
- `d2u/dx2`: Second derivative in x
- `du/dx`: First derivative in x
- `u`: Function value
- Constants: `pi`, `e`, numbers

### Boundary Conditions
- Dirichlet: `{"type": "dirichlet", "values": {"x=0": 0, "x=1": 0}}`
- Neumann: `{"type": "neumann", "values": {"x=0": "cos(pi*y)"}}`
- Mixed: `{"type": "mixed", "dirichlet": {...}, "neumann": {...}}`

## Device and Performance

### Hardware Recommendations
- **Neural networks**: Use GPU (`"device": "cuda"`) when available
- **Linear algebra**: Use CPU (`"linear_device": "cpu"`) for stability
- **Memory**: Monitor GPU memory usage, reduce parameters if needed

### Typical Parameters
- `epochs_adam`: 10000-30000 for initial training
- `hidden_dims`: [32, 64, 32] for 2D problems
- `n_segments`: [10, 10] for 2D domain decomposition
- `poly_degree`: [5, 5] for 2D polynomial features
- `points_domain`: 20000 for training points

## Version-Specific Notes

Current version: v0.2
- High-accuracy solving for arbitrary-dimensional linear PDEs
- Auto-coding capability for PDEs
- GPU acceleration support
- English documentation and comments