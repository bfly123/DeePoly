# DeePoly Program Structure Documentation

## Overview
DeePoly is a high-order accuracy neural network framework for function approximation and PDE solving using a hybrid "Scoper + Sniper" approach combining neural networks with polynomials.

## Project Architecture

```
DeePoly/
├── src/
│   ├── main_solver.py              # 🎯 Main entry point and workflow orchestration
│   │
│   ├── abstract_class/             # 🏗️ Core framework abstractions and base classes
│   │   ├── __init__.py
│   │   ├── base_fitter.py          # Base DeePoly fitting algorithm
│   │   ├── base_net.py             # Neural network base class with segment handling
│   │   ├── base_boundary.py        # Boundary condition abstractions
│   │   ├── operator_factory.py     # Differential operator generation factory
│   │   ├── features_generator.py   # Feature generation for neural networks
│   │   ├── boundary_constraint.py  # Boundary constraint implementations
│   │   ├── constants.py            # Mathematical and physical constants
│   │   ├── operator_usage_example.py  # Example usage of operators
│   │   └── config/                 # Configuration management
│   │       ├── __init__.py
│   │       ├── base_config.py      # Base configuration class
│   │       ├── base_data.py        # Data configuration abstractions
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
│   │   │       ├── config.py       # Linear PDE configuration
│   │   │       ├── data.py         # Linear PDE data handling
│   │   │       └── visualize.py    # Linear PDE visualization
│   │   │
│   │   ├── time_pde_solver/        # Time-dependent PDE problems
│   │   │   ├── __init__.py
│   │   │   ├── solver.py           # Main time PDE solver with IMEX schemes
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── net.py          # Neural network for time PDEs
│   │   │   │   └── fitter.py       # Time-stepping fitter implementation
│   │   │   ├── time_schemes/       # Time integration schemes
│   │   │   │   ├── __init__.py
│   │   │   │   ├── factory.py      # Time scheme factory
│   │   │   │   ├── base_time_scheme.py      # Base time integration class
│   │   │   │   ├── imex_1st.py     # First-order IMEX scheme
│   │   │   │   ├── imex_rk_222.py  # IMEX-RK(2,2,2) scheme
│   │   │   │   ├── onestep_predictor.py     # One-step predictor scheme
│   │   │   │   └── old_onestep.py  # Legacy one-step implementation
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── config.py       # Time PDE configuration
│   │   │       ├── data.py         # Time PDE data handling
│   │   │       └── visualize.py    # Time evolution visualization & animation
│   │   │
│   │   └── general_pde_solver/     # General PDE solver (placeholder)
│   │       └── __init__.py
│   │
│   ├── algebraic_solver/           # 🧮 Linear algebra backend systems
│   │   ├── __init__.py
│   │   ├── linear_solver.py        # GPU-accelerated linear system solver
│   │   ├── gauss_newton.py         # Gauss-Newton optimization
│   │   ├── fastnewton.py          # Fast Newton method
│   │   └── trustregionsolver.py   # Trust region optimization
│   │
│   └── meta_coding/                # 🤖 Automatic code generation utilities
│       ├── __init__.py
│       ├── auto_spotter.py         # Automatic PDE code generation
│       ├── auto_code_manager.py    # Auto-code workflow management
│       ├── auto_snipper.py         # Code snippet generation
│       └── auto_snipper_test.py    # Testing for auto snipper
│
├── cases/                          # 📁 Test cases and examples
│   ├── func_fitting_cases/
│   ├── linear_pde_cases/
│   └── time_pde_cases/
│
└── CLAUDE.md                       # Project documentation and guidelines
```

## Core Components Flow Diagram

```mermaid
graph TD
    A[main_solver.py] --> B[AutoCodeManager]
    A --> C[Problem Solver Factory]

    B --> D[auto_spotter.py]
    B --> E[Consistency Check]

    C --> F[FuncFittingSolver]
    C --> G[LinearPDESolver]
    C --> H[TimePDESolver]

    F --> I[base_fitter.py]
    G --> I
    H --> I

    I --> J[base_net.py]
    I --> K[operator_factory.py]
    I --> L[features_generator.py]

    H --> M[Time Schemes]
    M --> N[IMEX-RK(2,2,2)]
    M --> O[IMEX-1st]
    M --> P[OneStep Predictor]

    J --> Q[algebraic_solver]
    Q --> R[linear_solver.py]

    subgraph "Visualization Pipeline"
        S[visualize.py] --> T[Real-time Animation]
        S --> U[Error Analysis]
        S --> V[GIF Generation]
    end

    H --> S
```

## Dependency Hierarchy (Bottom-Up)

### Level 1: Core Utilities and Constants
```
abstract_class/constants.py
abstract_class/config/base_config.py
abstract_class/config/base_data.py
abstract_class/config/base_visualize.py
```

### Level 2: Mathematical Foundations
```
abstract_class/operator_factory.py
abstract_class/features_generator.py
abstract_class/boundary_constraint.py
abstract_class/base_boundary.py
algebraic_solver/linear_solver.py
algebraic_solver/gauss_newton.py
algebraic_solver/fastnewton.py
algebraic_solver/trustregionsolver.py
```

### Level 3: Neural Network Abstractions
```
abstract_class/base_net.py
abstract_class/base_fitter.py
```

### Level 4: Problem-Specific Configurations
```
problem_solvers/func_fitting_solver/utils/config.py
problem_solvers/linear_pde_solver/utils/config.py
problem_solvers/time_pde_solver/utils/config.py
problem_solvers/*/utils/data.py
```

### Level 5: Core Problem Implementations
```
problem_solvers/func_fitting_solver/core/net.py
problem_solvers/func_fitting_solver/core/fitter.py
problem_solvers/linear_pde_solver/core/net.py
problem_solvers/linear_pde_solver/core/fitter.py
problem_solvers/time_pde_solver/core/net.py
problem_solvers/time_pde_solver/core/fitter.py
```

### Level 6: Time Integration Schemes
```
problem_solvers/time_pde_solver/time_schemes/base_time_scheme.py
problem_solvers/time_pde_solver/time_schemes/imex_1st.py
problem_solvers/time_pde_solver/time_schemes/imex_rk_222.py
problem_solvers/time_pde_solver/time_schemes/onestep_predictor.py
problem_solvers/time_pde_solver/time_schemes/factory.py
```

### Level 7: Visualization and Analysis
```
problem_solvers/*/utils/visualize.py
```

### Level 8: Main Solver Implementations
```
problem_solvers/func_fitting_solver/solver.py
problem_solvers/linear_pde_solver/solver.py
problem_solvers/time_pde_solver/solver.py
```

### Level 9: Meta-Coding and Automation
```
meta_coding/auto_spotter.py
meta_coding/auto_snipper.py
meta_coding/auto_code_manager.py
```

### Level 10: Top-Level Orchestration
```
main_solver.py
```

## Key Design Patterns

### 1. **Factory Pattern**
- `operator_factory.py`: Creates differential operators based on PDE specifications
- `time_schemes/factory.py`: Creates time integration schemes
- Problem solver selection in `main_solver.py`

### 2. **Template Method Pattern**
- `base_fitter.py`: Defines the general fitting algorithm workflow
- `base_net.py`: Provides common neural network operations
- `base_time_scheme.py`: Template for time integration methods

### 3. **Strategy Pattern**
- Time integration schemes (IMEX-1st, IMEX-RK-222, OneStep)
- Linear algebra solvers (SVD, QR, GPU-accelerated)
- Boundary condition implementations

### 4. **Observer Pattern**
- Real-time visualization updates during time stepping
- Error monitoring and statistics collection

## Data Flow Architecture

### Input Processing
```
config.json → Config Classes → Operator Factory → Mathematical Operators
```

### Core Computation Loop
```
Initial Conditions → Neural Network Training → Linear System Solve → Solution Update
```

### Time Evolution (Time PDEs)
```
Time Step → IMEX Scheme → L1/L2/F/N Operators → Neural Features → Polynomial Coefficients
```

### Output Generation
```
Solution Data → Visualization → Error Analysis → Reports & Animations
```

## Module Responsibilities

| Module | Primary Responsibility | Key Features |
|--------|----------------------|--------------|
| `main_solver.py` | Workflow orchestration | Auto-code detection, solver selection, process management |
| `abstract_class/` | Framework foundations | Base classes, operator generation, feature extraction |
| `problem_solvers/` | Domain-specific solving | Function fitting, linear PDEs, time-dependent PDEs |
| `algebraic_solver/` | Linear algebra backend | GPU acceleration, optimization algorithms |
| `meta_coding/` | Code generation | Automatic PDE operator generation, consistency checking |
| `time_schemes/` | Time integration | IMEX methods, predictor schemes |
| `utils/` | Support utilities | Configuration, data handling, visualization |

## Critical Integration Points

1. **Config → Operator Pipeline**: Configuration files drive operator generation
2. **Neural → Algebraic Interface**: Neural features feed into linear algebra solvers
3. **Time → Space Coupling**: Time schemes coordinate with spatial discretization
4. **Auto-Code Consistency**: Automatic detection and regeneration of outdated code
5. **Visualization Hooks**: Real-time updates during computation

This architecture supports the hybrid "Scoper + Sniper" approach where neural networks provide global approximation (Scoper) and polynomials refine for high-order accuracy (Sniper).