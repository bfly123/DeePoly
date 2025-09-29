# DeePoly Program Structure Documentation

## Overview

DeePoly is a high-order accuracy neural network framework for function approximation and PDE solving that provides unified configuration, unified boundary condition processing, automatic code generation, and IMEX time integration schemes for robust nonlinear equation solving.

## Project Architecture

```
DeePoly/
├── src/
│   ├── main_solver.py              # 🎯 Main entry point with unified factory and AutoCode workflow
│   │
│   ├── abstract_class/             # 🏗️ Core abstractions and unified base classes
│   │   ├── __init__.py
│   │   ├── base_fitter.py          # Base DeePoly fitting algorithm with unified constraint processing
│   │   ├── base_net.py             # Neural network base class with segment handling
│   │   ├── operator_factory.py     # Differential operator generation factory
│   │   ├── features_generator.py   # Feature generation for neural networks
│   │   ├── boundary_conditions.py  # ✨ Unified boundary condition processing + Mixin
│   │   ├── boundary_constraint.py  # Boundary constraint implementations (Dirichlet/Neumann/Robin/Periodic)
│   │   ├── constants.py            # Mathematical and physical constants
│   │   ├── operator_usage_example.py  # Example usage of operators
│   │   └── config/                 # 🚀 Unified configuration management
│   │       ├── __init__.py
│   │       ├── base_pde_config.py  # ✨ Unified PDE configuration (eq={L1,L2,F,N,S} standardization)
│   │       ├── base_config.py      # Base configuration class
│   │       ├── base_data.py        # Unified data pipeline/boundary/source/reference solution loading
│   │       └── base_visualize.py   # Visualization base class
│   │
│   ├── problem_solvers/            # 🔧 Problem-specific solver implementations
│   │   ├── __init__.py
│   │   │
│   │   ├── func_fitting_solver/    # Function approximation problems
│   │   │   ├── __init__.py
│   │   │   ├── solver.py           # Main function fitting solver
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── net.py          # Neural network for function fitting
│   │   │   │   └── fitter.py       # Fitting algorithm implementation
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── config.py       # Function fitting configuration
│   │   │       ├── data.py         # Data generation and handling
│   │   │       └── visualize.py    # Function fitting visualization
│   │   │
│   │   ├── linear_pde_solver/      # Linear PDE problems
│   │   │   ├── __init__.py
│   │   │   ├── solver.py           # Main linear PDE solver
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── net.py          # Neural network for linear PDEs
│   │   │   │   └── fitter.py       # Linear PDE fitting algorithm
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── config.py       # Linear PDE configuration (inherits from BasePDEConfig)
│   │   │       ├── data.py         # Linear PDE data handling
│   │   │       └── visualize.py    # Linear PDE visualization
│   │   │
│   │   ├── time_pde_solver/        # ⏱️ Time-dependent PDE problems
│   │   │   ├── __init__.py
│   │   │   ├── solver.py           # Main time PDE solver with IMEX schemes
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── net.py          # Neural network for time PDEs
│   │   │   │   └── fitter.py       # Time-stepping fitter implementation
│   │   │   ├── time_schemes/       # ⚡ Time integration schemes
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_time_scheme.py      # Base time integration class
│   │   │   │   ├── imex_1st.py     # First-order IMEX scheme
│   │   │   │   ├── imex_rk_222.py  # ✨ IMEX-RK(2,2,2) scheme
│   │   │   │   ├── onestep_predictor.py     # One-step predictor scheme
│   │   │   │   └── factory.py      # Time scheme factory registration
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── config.py       # Time PDE configuration (inherits from BasePDEConfig)
│   │   │       ├── data.py         # Time PDE data handling
│   │   │       └── visualize.py    # Time evolution visualization & animation
│   │   │
│   │   └── general_pde_solver/     # General PDE solver (placeholder)
│   │       └── __init__.py
│   │
│   ├── algebraic_solver/           # 🧮 Enhanced nonlinear and linear algebra backend
│   │   ├── __init__.py
│   │   ├── linear_solver.py        # GPU-accelerated linear system solver (CPU/GPU, SVD/QR/parallel/mixed precision)
│   │   ├── gauss_newton.py         # ✨ LM damping + Wolfe line search + PCG fallback
│   │   ├── trustregionsolver.py    # ✨ Preconditioned trust region (CPU/GPU)
│   │   └── fastnewton.py          # ✨ Fast Newton method (CPU/GPU)
│   │
│   └── meta_coding/                # 🤖 Automatic code generation and maintenance
│       ├── __init__.py
│       ├── auto_spotter.py         # ✨ Automatic generation/update of net.py auto code segments
│       ├── auto_code_manager.py    # ✨ Consistency checking, auto-generation, and restart workflow
│       ├── auto_snipper.py         # Code snippet generation
│       ├── auto_eq.py              # Equation processing
│       └── auto_repalce_nonlinear.py # Nonlinear replacement utilities
│
├── cases/                          # 📁 Test cases and examples
│   ├── func_fitting_cases/
│   ├── linear_pde_cases/
│   │   ├── poisson_2d_sinpixsinpiy # Poisson 2D example
│   │   └── test_file_source        # Source term file loading example
│   └── Time_pde_cases/
│       └── KDV_equation            # KdV time PDE example
│
├── Doc/                            # 📚 Documentation
│   ├── boundary_condition_unification_analysis.md
│   ├── imexRK.md / imex_rk_222_optimization.md
│   ├── time_scheme_program.md
│   ├── Config_JSON_Periodic_BC_Simplification.md
│   ├── archived/                   # Archived Chinese documentation
│   └── logo_deepoly.png
│
├── CLAUDE.md                       # Project guide for Claude Code
├── README.md                       # Main project documentation
└── PROGRAM_STRUCTURE.md           # This file
```

## Core Flow Architecture

### Main Workflow
```
main_solver.py: Parse --case_path → AutoCodeManager consistency check/trigger generation →
Read config.json → Factory create corresponding solver → Solve and output
```

### Unified Solver Pattern
All solvers follow this unified architecture:

1. **Config**: Inherits from `BasePDEConfig` with unified eq semantics, segment division, operator parsing
2. **Data**: Inherits from `BaseDataGenerator` with unified point sets, source terms, reference solutions, boundary data
3. **Net**: `BaseNet` + `BoundaryConditionMixin`, with `physics_loss` containing `# auto code` segments
4. **Fitter**: `BaseDeepPolyFitter` precompiles operators/features, delegates linear subproblems to `LinearSolver`, with IMEX time schemes for time PDEs

### Boundary Unification
- `boundary_conditions.py` provides unified processor/validation and GPU acceleration
- `BoundaryConditionMixin` injected into all Net classes
- Unified handling for Dirichlet/Neumann/Robin/Periodic constraints

### Automatic Code Generation
- `auto_spotter.py` reads configuration to generate/replace `# auto code begin/end` segments in net.py
- `auto_code_manager.py` manages generation and signature verification for consistency

## Dependency Hierarchy (Bottom-Up)

### Level 1: Core and Configuration
```
abstract_class/config/base_pde_config.py  # ✨ Unified PDE configuration
abstract_class/config/base_config.py
abstract_class/config/base_data.py
abstract_class/config/base_visualize.py
abstract_class/constants.py
```

### Level 2: Mathematical/Feature/Boundary Foundations
```
abstract_class/operator_factory.py
abstract_class/features_generator.py
abstract_class/boundary_conditions.py     # ✨ Unified boundary processing
abstract_class/boundary_constraint.py
```

### Level 3: Algebraic and Optimization
```
algebraic_solver/linear_solver.py
algebraic_solver/gauss_newton.py         # ✨ Enhanced nonlinear solving
algebraic_solver/trustregionsolver.py    # ✨ Trust region methods
algebraic_solver/fastnewton.py          # ✨ Fast Newton methods
```

### Level 4: Framework Base Classes
```
abstract_class/base_net.py               # With BoundaryConditionMixin
abstract_class/base_fitter.py           # Unified constraint processing
```

### Level 5: Problem-Specific Configurations and Utils
```
problem_solvers/func_fitting_solver/utils/{config.py,data.py,visualize.py}
problem_solvers/linear_pde_solver/utils/{config.py,data.py,visualize.py}
problem_solvers/time_pde_solver/utils/{config.py,data.py,visualize.py}
```

### Level 6: Core Problem Implementations
```
problem_solvers/func_fitting_solver/core/{net.py,fitter.py}
problem_solvers/linear_pde_solver/core/{net.py,fitter.py}
problem_solvers/time_pde_solver/core/{net.py,fitter.py}
```

### Level 7: Time Integration Schemes
```
problem_solvers/time_pde_solver/time_schemes/base_time_scheme.py
problem_solvers/time_pde_solver/time_schemes/imex_1st.py
problem_solvers/time_pde_solver/time_schemes/imex_rk_222.py  # ✨ Advanced IMEX
problem_solvers/time_pde_solver/time_schemes/onestep_predictor.py
problem_solvers/time_pde_solver/time_schemes/factory.py     # ✨ Factory registration
```

### Level 8: Main Solver Implementations
```
problem_solvers/func_fitting_solver/solver.py
problem_solvers/linear_pde_solver/solver.py
problem_solvers/time_pde_solver/solver.py
```

### Level 9: Meta-Coding and Automation
```
meta_coding/auto_spotter.py            # ✨ Auto code generation
meta_coding/auto_code_manager.py       # ✨ Consistency management
meta_coding/auto_snipper.py
meta_coding/auto_eq.py
meta_coding/auto_repalce_nonlinear.py
```

### Level 10: Top-Level Orchestration
```
main_solver.py                         # ✨ Unified factory and workflow
```

## New Version Key Improvements

### 1. **Enhanced Nonlinear Equation Solving** (`src/algebraic_solver/*`)
- **LM Damping + Wolfe Line Search**: `gauss_newton.py` with enhanced stability and convergence
- **Preconditioned Trust Region**: `trustregionsolver.py` with CPU/GPU support
- **Fast Newton Methods**: `fastnewton.py` with optimized CPU/GPU implementations
- **Adaptive Solvers**: Automatic fallback between methods for robust convergence

### 2. **Unified Codebase Maintenance and Organization**
- **Unified Configuration**: `BasePDEConfig` provides consistent `eq={L1,L2,F,N,S}` semantics across all solvers
- **Relaxed Loading**: Backward compatibility with legacy configuration formats
- **Unified Factory**: `main_solver.py` provides single entry point with consistent solver selection
- **Dimension Standardization**: Consistent tensor operations and shape management

### 3. **Boundary Condition Unification** (`src/abstract_class/boundary_conditions.py`)
- **Unified Data Structure**: Consistent boundary condition specification across all solvers
- **GPU Acceleration**: Efficient GPU-based boundary condition processing
- **BoundaryConditionMixin**: Reusable boundary processing for all neural networks
- **Periodic Boundary Simplification**: Unified paired constraint implementation

### 4. **Time PDE IMEX Schemes** (`src/problem_solvers/time_pde_solver/time_schemes/`)
- **IMEX-RK(2,2,2)**: Advanced implicit-explicit Runge-Kutta method
- **First-order IMEX**: Stable first-order time integration
- **One-step Predictor**: Efficient single-step prediction schemes
- **Factory Registration**: Pluggable time scheme architecture

### 5. **Configuration System Unification**
- **Standardized Format**: `eq={L1,L2,F,N,S}` format across all problem types
- **Backward Compatibility**: Seamless migration from legacy formats
- **Source Term Flexibility**: Support for expressions, files, and arrays
- **Reference Solution Support**: Multiple formats (.mat files, .py functions, expressions)

### 6. **Automatic Code Generation and Maintenance**
- **Auto Code Generation**: `auto_spotter.py` automatically generates/updates `# auto code` segments in net.py
- **Consistency Verification**: `auto_code_manager.py` performs signature checking and triggers regeneration
- **One-time Restart**: Automatic detection and regeneration of outdated code
- **Configuration-driven**: Code generation directly from `config.json` specifications

## Data Flow Architecture

### Input Processing
```
config.json → BasePDEConfig → Operator Factory → Mathematical Operators
```

### Core Computation Loop
```
Initial Conditions → Neural Network Training → Enhanced Nonlinear/Linear Solve → Solution Update
```

### Time Evolution (Time PDEs)
```
Time Step → IMEX Scheme → L1/L2/F/N Operators → Neural Features → Polynomial Coefficients
```

### Boundary Processing
```
Boundary Conditions → BoundaryConditionMixin → GPU Processing → Constraint Integration
```

### Output Generation
```
Solution Data → Unified Visualization → Error Analysis → Reports & Animations
```

## Module Responsibilities

| Module | Primary Responsibility | Key New Features |
|--------|----------------------|------------------|
| `main_solver.py` | Unified workflow orchestration | Auto-code detection, factory pattern, process management |
| `abstract_class/config/` | Unified configuration management | BasePDEConfig, relaxed loading, format standardization |
| `abstract_class/boundary_conditions.py` | Unified boundary processing | GPU acceleration, BoundaryConditionMixin, simplified periodic |
| `problem_solvers/` | Domain-specific solving | Consistent architecture, unified configuration inheritance |
| `algebraic_solver/` | Enhanced nonlinear/linear algebra | LM damping, trust region, fast Newton, CPU/GPU support |
| `time_schemes/` | Advanced time integration | IMEX-RK(2,2,2), factory registration, pluggable architecture |
| `meta_coding/` | Automatic code maintenance | Code generation, consistency checking, configuration-driven updates |

## Critical Integration Points

1. **Unified Config → Operator Pipeline**: `BasePDEConfig` drives consistent operator generation across all solvers
2. **Enhanced Neural → Algebraic Interface**: Neural features feed into advanced nonlinear/linear solvers with CPU/GPU optimization
3. **Time → Space Coupling**: IMEX schemes coordinate with spatial discretization and boundary processing
4. **Auto-Code Consistency**: Automatic detection, generation, and verification of code segments
5. **Boundary Unification**: Consistent boundary processing with GPU acceleration across all problem types
6. **Factory Pattern**: Unified solver selection and time scheme registration

This enhanced architecture maintains the hybrid "Scoper + Sniper" approach while adding robust nonlinear solving capabilities, unified system maintenance, and advanced time integration methods for industrial-strength PDE solving.