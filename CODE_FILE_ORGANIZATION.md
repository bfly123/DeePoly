# DeePoly Code File Organization

## File Organization by Dependency Order

This document organizes all code files in dependency order, from foundational utilities to top-level orchestration, ensuring proper understanding of the codebase structure.

---

## Level 1: Core Constants and Base Configurations

### 1.1 Mathematical Constants
**File**: `src/abstract_class/constants.py`
- **Purpose**: Defines mathematical and physical constants used throughout the framework
- **Dependencies**: None (foundational)
- **Key Components**: Mathematical constants, tolerance values, default parameters

### 1.2 Base Configuration System
**File**: `src/abstract_class/config/base_config.py`
- **Purpose**: Base configuration class with common configuration functionality
- **Dependencies**: `constants.py`
- **Key Components**: Configuration loading, validation, default values

**File**: `src/abstract_class/config/base_data.py`
- **Purpose**: Abstract data configuration and management
- **Dependencies**: `base_config.py`
- **Key Components**: Data loading interfaces, preprocessing abstractions

**File**: `src/abstract_class/config/base_visualize.py`
- **Purpose**: Base visualization configuration and utilities
- **Dependencies**: `base_config.py`
- **Key Components**: Plotting parameters, visualization settings

---

## Level 2: Mathematical Foundations

### 2.1 Operator Generation
**File**: `src/abstract_class/operator_factory.py`
- **Purpose**: Factory for generating differential operators from symbolic expressions
- **Dependencies**: `constants.py`
- **Key Components**:
  - Symbolic differentiation parsing
  - Operator term generation
  - Coefficient extraction
  - Matrix operator construction

### 2.2 Feature Generation
**File**: `src/abstract_class/features_generator.py`
- **Purpose**: Generates neural network features combining polynomial and neural components
- **Dependencies**: `constants.py`, `operator_factory.py`
- **Key Components**:
  - Polynomial feature generation
  - Neural feature extraction
  - Feature combination strategies
  - Segment-wise feature handling

### 2.3 Boundary Constraints
**File**: `src/abstract_class/boundary_constraint.py`
- **Purpose**: Implementation of various boundary condition types
- **Dependencies**: `constants.py`
- **Key Components**:
  - Dirichlet boundary conditions
  - Neumann boundary conditions
  - Robin boundary conditions
  - Periodic boundary conditions

**File**: `src/abstract_class/base_boundary.py`
- **Purpose**: Abstract base class for boundary condition handling
- **Dependencies**: `boundary_constraint.py`
- **Key Components**: Boundary constraint interface, validation methods

### 2.4 Linear Algebra Backend
**File**: `src/algebraic_solver/linear_solver.py`
- **Purpose**: GPU-accelerated linear system solver with multiple methods
- **Dependencies**: External libraries (NumPy, CuPy, SciPy)
- **Key Components**:
  - SVD solver
  - QR decomposition
  - GPU acceleration support
  - Condition number analysis
  - Residual analysis

**File**: `src/algebraic_solver/gauss_newton.py`
- **Purpose**: Gauss-Newton optimization algorithm implementation
- **Dependencies**: `linear_solver.py`
- **Key Components**: Iterative optimization, Jacobian computation

**File**: `src/algebraic_solver/fastnewton.py`
- **Purpose**: Fast Newton method for nonlinear optimization
- **Dependencies**: `linear_solver.py`
- **Key Components**: Accelerated Newton iterations, convergence criteria

**File**: `src/algebraic_solver/trustregionsolver.py`
- **Purpose**: Trust region optimization methods
- **Dependencies**: `linear_solver.py`
- **Key Components**: Trust region updates, step acceptance criteria

---

## Level 3: Neural Network Abstractions

### 3.1 Base Neural Network
**File**: `src/abstract_class/base_net.py`
- **Purpose**: Base class for neural networks with segment handling and boundary conditions
- **Dependencies**: `features_generator.py`, `base_boundary.py`, `linear_solver.py`
- **Key Components**:
  - Multi-segment neural network architecture
  - Boundary condition enforcement
  - Feature extraction and combination
  - Forward pass implementation
  - Loss computation

### 3.2 Base Fitting Algorithm
**File**: `src/abstract_class/base_fitter.py`
- **Purpose**: Base class implementing the core DeePoly fitting algorithm
- **Dependencies**: `base_net.py`, `operator_factory.py`, `algebraic_solver/`
- **Key Components**:
  - Hybrid neural-polynomial fitting workflow
  - Training loop management
  - Convergence criteria
  - Solution reconstruction
  - Error analysis

---

## Level 4: Problem-Specific Configurations

### 4.1 Function Fitting Configuration
**File**: `src/problem_solvers/func_fitting_solver/utils/config.py`
- **Purpose**: Configuration class for function approximation problems
- **Dependencies**: `abstract_class/config/base_config.py`
- **Key Components**: Function fitting parameters, domain specification

**File**: `src/problem_solvers/func_fitting_solver/utils/data.py`
- **Purpose**: Data generation and handling for function fitting
- **Dependencies**: `base_data.py`, function fitting config
- **Key Components**: Training data generation, validation data setup

### 4.2 Linear PDE Configuration
**File**: `src/problem_solvers/linear_pde_solver/utils/config.py`
- **Purpose**: Configuration class for linear PDE problems
- **Dependencies**: `abstract_class/config/base_config.py`
- **Key Components**: PDE specification, boundary conditions, domain discretization

**File**: `src/problem_solvers/linear_pde_solver/utils/data.py`
- **Purpose**: Data handling for linear PDE problems
- **Dependencies**: `base_data.py`, linear PDE config
- **Key Components**: Grid generation, boundary data setup

### 4.3 Time PDE Configuration
**File**: `src/problem_solvers/time_pde_solver/utils/config.py`
- **Purpose**: Configuration class for time-dependent PDE problems
- **Dependencies**: `abstract_class/config/base_config.py`
- **Key Components**:
  - Time integration parameters
  - IMEX operator splitting
  - Reference solution handling
  - Animation settings

**File**: `src/problem_solvers/time_pde_solver/utils/data.py`
- **Purpose**: Data handling for time-dependent PDEs
- **Dependencies**: `base_data.py`, time PDE config
- **Key Components**: Initial conditions, time grid setup, reference data loading

---

## Level 5: Core Problem Implementations

### 5.1 Function Fitting Core
**File**: `src/problem_solvers/func_fitting_solver/core/net.py`
- **Purpose**: Neural network specialized for function approximation
- **Dependencies**: `abstract_class/base_net.py`
- **Key Components**: Function-specific network architecture, loss functions

**File**: `src/problem_solvers/func_fitting_solver/core/fitter.py`
- **Purpose**: Fitting algorithm for function approximation
- **Dependencies**: `abstract_class/base_fitter.py`, function fitting core net
- **Key Components**: Function fitting workflow, approximation error analysis

### 5.2 Linear PDE Core
**File**: `src/problem_solvers/linear_pde_solver/core/net.py`
- **Purpose**: Neural network for linear PDE problems
- **Dependencies**: `abstract_class/base_net.py`
- **Key Components**: PDE-specific boundary handling, residual computation

**File**: `src/problem_solvers/linear_pde_solver/core/fitter.py`
- **Purpose**: Fitting algorithm for linear PDEs
- **Dependencies**: `abstract_class/base_fitter.py`, linear PDE core net
- **Key Components**: Linear PDE solving workflow, residual minimization

### 5.3 Time PDE Core
**File**: `src/problem_solvers/time_pde_solver/core/net.py`
- **Purpose**: Neural network for time-dependent PDEs with boundary conditions
- **Dependencies**: `abstract_class/base_net.py`
- **Key Components**:
  - Time-dependent boundary condition enforcement
  - Periodic boundary condition handling
  - Cross-segment constraints
  - IMEX operator application

**File**: `src/problem_solvers/time_pde_solver/core/fitter.py`
- **Purpose**: Time-stepping fitting algorithm
- **Dependencies**: `abstract_class/base_fitter.py`, time PDE core net
- **Key Components**:
  - Time step execution
  - Neural network parameter inheritance
  - Spotter skip optimization
  - Solution update mechanism

---

## Level 6: Time Integration Schemes

### 6.1 Base Time Integration
**File**: `src/problem_solvers/time_pde_solver/time_schemes/base_time_scheme.py`
- **Purpose**: Abstract base class for all time integration schemes
- **Dependencies**: Core fitter classes
- **Key Components**: Time step interface, operator splitting framework

### 6.2 IMEX Schemes
**File**: `src/problem_solvers/time_pde_solver/time_schemes/imex_1st.py`
- **Purpose**: First-order IMEX (Implicit-Explicit) time integration
- **Dependencies**: `base_time_scheme.py`
- **Key Components**: L1 implicit treatment, L2/F/N explicit treatment

**File**: `src/problem_solvers/time_pde_solver/time_schemes/imex_rk_222.py`
- **Purpose**: Second-order IMEX Runge-Kutta (2,2,2) scheme
- **Dependencies**: `base_time_scheme.py`
- **Key Components**: Multi-stage RK integration, higher-order accuracy

### 6.3 Predictor Schemes
**File**: `src/problem_solvers/time_pde_solver/time_schemes/onestep_predictor.py`
- **Purpose**: One-step predictor time integration scheme
- **Dependencies**: `base_time_scheme.py`
- **Key Components**: Adaptive time stepping, error control

**File**: `src/problem_solvers/time_pde_solver/time_schemes/old_onestep.py`
- **Purpose**: Legacy one-step implementation (deprecated)
- **Dependencies**: `base_time_scheme.py`
- **Key Components**: Original one-step algorithm

### 6.4 Time Scheme Factory
**File**: `src/problem_solvers/time_pde_solver/time_schemes/factory.py`
- **Purpose**: Factory for creating time integration schemes
- **Dependencies**: All time scheme implementations
- **Key Components**: Scheme selection logic, parameter validation

---

## Level 7: Visualization and Analysis

### 7.1 Function Fitting Visualization
**File**: `src/problem_solvers/func_fitting_solver/utils/visualize.py`
- **Purpose**: Visualization tools for function approximation results
- **Dependencies**: `base_visualize.py`, function fitting solver
- **Key Components**: Function plots, error visualization, convergence analysis

### 7.2 Linear PDE Visualization
**File**: `src/problem_solvers/linear_pde_solver/utils/visualize.py`
- **Purpose**: Visualization tools for linear PDE solutions
- **Dependencies**: `base_visualize.py`, linear PDE solver
- **Key Components**: 2D/3D solution plots, contour plots, error analysis

### 7.3 Time PDE Visualization
**File**: `src/problem_solvers/time_pde_solver/utils/visualize.py`
- **Purpose**: Advanced visualization for time-dependent PDEs
- **Dependencies**: `base_visualize.py`, time PDE solver
- **Key Components**:
  - Real-time animation during computation
  - Time evolution plotting
  - Error statistics and reporting
  - GIF animation generation
  - Reference solution comparison
  - Spacetime visualization

---

## Level 8: Main Solver Implementations

### 8.1 Function Fitting Solver
**File**: `src/problem_solvers/func_fitting_solver/solver.py`
- **Purpose**: Main solver for function approximation problems
- **Dependencies**: Function fitting core, utils, config
- **Key Components**: Solver orchestration, result analysis

### 8.2 Linear PDE Solver
**File**: `src/problem_solvers/linear_pde_solver/solver.py`
- **Purpose**: Main solver for linear PDE problems
- **Dependencies**: Linear PDE core, utils, config
- **Key Components**: Linear PDE solving workflow, convergence monitoring

### 8.3 Time PDE Solver
**File**: `src/problem_solvers/time_pde_solver/solver.py`
- **Purpose**: Main solver for time-dependent PDE problems
- **Dependencies**: Time PDE core, time schemes, utils, config
- **Key Components**:
  - Time evolution loop management
  - Reference solution loading and comparison
  - Real-time visualization coordination
  - Error analysis and reporting
  - Performance timing
  - Result export

---

## Level 9: Meta-Coding and Automation

### 9.1 Automatic Code Generation
**File**: `src/meta_coding/auto_spotter.py`
- **Purpose**: Automatic generation of PDE operator code
- **Dependencies**: `operator_factory.py`, problem solver configurations
- **Key Components**:
  - Config signature generation
  - Operator code template generation
  - File writing and backup management
  - Consistency verification

### 9.2 Code Snippet Management
**File**: `src/meta_coding/auto_snipper.py`
- **Purpose**: Automatic code snippet generation utilities
- **Dependencies**: Auto spotter utilities
- **Key Components**: Code template management, snippet generation

**File**: `src/meta_coding/auto_snipper_test.py`
- **Purpose**: Testing utilities for auto snipper functionality
- **Dependencies**: `auto_snipper.py`
- **Key Components**: Unit tests, validation functions

### 9.3 Auto-Code Workflow Management
**File**: `src/meta_coding/auto_code_manager.py`
- **Purpose**: Manages the complete auto-code generation workflow
- **Dependencies**: `auto_spotter.py`, all solver configurations
- **Key Components**:
  - Consistency checking between config and generated code
  - Code generation triggering
  - Process restart management
  - Error handling and recovery

---

## Level 10: Top-Level Orchestration

### 10.1 Main Entry Point
**File**: `src/main_solver.py`
- **Purpose**: Main entry point and workflow orchestration
- **Dependencies**: All solver implementations, auto-code manager
- **Key Components**:
  - Command-line argument parsing
  - Case directory validation
  - Auto-code consistency checking
  - Solver selection and instantiation
  - Error handling and logging
  - Process lifecycle management

---

## Usage Examples and Documentation

### Example Usage File
**File**: `src/abstract_class/operator_usage_example.py`
- **Purpose**: Demonstrates how to use the operator factory system
- **Dependencies**: `operator_factory.py`
- **Key Components**: Usage examples, best practices, common patterns

---

## Module Initialization Files

All `__init__.py` files serve to:
- Define package structure
- Export public interfaces
- Handle module-level imports
- Provide package documentation

### Key Init Files:
- `src/__init__.py`: Root package initialization
- `src/abstract_class/__init__.py`: Core abstractions export
- `src/problem_solvers/__init__.py`: Solver factory exports
- `src/algebraic_solver/__init__.py`: Linear algebra exports
- `src/meta_coding/__init__.py`: Auto-coding utilities export

---

## Compilation and Execution Order

When building or understanding the system, follow this order:

1. **Foundation**: Constants → Base configs → Mathematical operators
2. **Core Framework**: Features → Boundaries → Neural nets → Fitting algorithms
3. **Problem-Specific**: Configs → Data → Core implementations
4. **Integration**: Time schemes → Visualization → Main solvers
5. **Automation**: Auto-coding → Workflow management
6. **Orchestration**: Main solver entry point

This organization ensures that dependencies are resolved in the correct order and provides a clear understanding of the system architecture from bottom to top.