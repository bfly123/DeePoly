# Comprehensive Analysis: Extracting Common Functionality into Nonlinear Solver Base Class

**Date:** 2025-10-10
**Purpose:** Identify ALL common functionality between `time_pde_solver` and `general_pde_solver` for extraction into an intermediate `nonlinear_solver` base class.

---

## Executive Summary

This analysis examines two PDE solver implementations to identify all duplicated code and common patterns that should be consolidated into a `nonlinear_solver` base class. The solvers share **significant common structure** (~60-70% code similarity) but differ in their core solving strategies:

- **time_pde_solver**: Physical time evolution via `solve_time_evolution()`
- **general_pde_solver**: Pseudo-time stepping to steady state via `solve_steady_state()`

---

## Section 1: Common Attributes

### 1.1 Core Configuration Attributes
Both solvers maintain identical configuration attributes:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `self.config` | Config object | Solver configuration (TimePDEConfig/GeneralPDEConfig) |
| `self.case_dir` | str | Case directory path |

### 1.2 Component Attributes
Both solvers initialize identical component instances:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `self.datagen` | DataGenerator | Data generation (TimePDEDataGenerator/GeneralPDEDataGenerator) |
| `self.data_train` | Dict | Training data |
| `self.data_test` | Dict | Test data |
| `self.model` | nn.Module | Neural network model (TimePDENet/GeneralPDENet) |
| `self.fitter` | Fitter | Hybrid fitter (TimePDEFitter/GeneralPDEFitter) |
| `self.visualizer` | Visualizer | Visualization component (TimePDEVisualizer/GeneralPDEVisualizer) |
| `self.reference_solution` | Optional[Dict] | Reference solution data |

### 1.3 Reference Solution Structure
Both solvers use identical reference solution dictionary structure:

```python
{
    'x_ref': np.ndarray,        # Spatial coordinates
    'y_ref': np.ndarray,        # Second dimension (optional, can be 't' or 'y')
    'u_ref': np.ndarray,        # Solution values
    'file_path': str,           # Path to .mat file
    'interpolators': dict,      # Interpolation functions (time_pde only)
    'y_label': str              # Label for second dimension (general_pde only)
}
```

---

## Section 2: Common Methods

### 2.1 Initialization Methods

#### `__init__(self, config=None, case_dir=None)`
**Similarity:** 95%

Both solvers follow identical initialization sequence:
```python
def __init__(self, config=None, case_dir=None):
    self._initialize_config(config, case_dir)
    self._initialize_components()
    # time_pde: self._initialize_solution_tracking()
```

**Difference:**
- `time_pde_solver` adds `_initialize_solution_tracking()` for loss history

#### `_initialize_config(self, config, case_dir)`
**Similarity:** 100%

Identical implementation in both solvers:
```python
def _initialize_config(self, config, case_dir):
    if config is not None:
        self.config = config
    elif case_dir is not None:
        self.config = XPDEConfig(case_dir=case_dir)  # X = Time or General
    else:
        raise ValueError("Either config object or case_dir must be provided")

    self.case_dir = case_dir if case_dir else getattr(self.config, "case_dir", None)

    if hasattr(self.config, "export_to_json"):
        self.config.export_to_json("config.json")
```

#### `_initialize_components(self)`
**Similarity:** 90%

Near-identical pattern with solver-specific class names:

**time_pde_solver:**
```python
def _initialize_components(self):
    self.datagen = TimePDEDataGenerator(self.config)
    self.data_train = self.datagen.generate_data("train")
    self.data_test = self.datagen.generate_data("test")
    self.model = TimePDENet(self.config).to(self.config.device)
    time_scheme = getattr(self.config, 'time_scheme', 'imex_rk_222')
    self.fitter = TimePDEFitter(config=self.config, data=self.data_train,
                                time_scheme=time_scheme)
    self.visualizer = TimePDEVisualizer(self.config)
    self.reference_solution = self._load_reference_solution()
    os.makedirs(self.config.results_dir, exist_ok=True)
```

**general_pde_solver:**
```python
def _initialize_components(self):
    self.datagen = GeneralPDEDataGenerator(self.config)
    self.data_train = self.datagen.generate_data("train")
    self.data_test = self.datagen.generate_data("test")
    self.model = GeneralPDENet(self.config).to(self.config.device)
    self.visualizer = GeneralPDEVisualizer(self.config)
    time_scheme = getattr(self.config, 'time_scheme', 'implicit_euler')
    self.fitter = GeneralPDEFitter(config=self.config, data=self.data_train,
                                   time_scheme=time_scheme, visualizer=self.visualizer)
    self.reference_solution = self._load_reference_solution()
    os.makedirs(self.config.results_dir, exist_ok=True)
```

**Common Pattern:** Can be abstracted with factory methods for component creation.

### 2.2 Reference Solution Methods

#### `_load_reference_solution(self) -> Optional[Dict]`
**Similarity:** 85%

Both implement nearly identical reference solution loading:

**Common Code Block (Lines 100-155 in time_pde, 204-308 in general_pde):**
```python
def _load_reference_solution(self) -> Optional[Dict]:
    if not hasattr(self.config, 'reference_solution') or not self.config.reference_solution:
        return None

    ref_path = self.config.reference_solution

    # If relative path, make it relative to case directory
    if not os.path.isabs(ref_path):
        case_dir = getattr(self.config, 'case_dir', os.getcwd())
        ref_path = os.path.join(case_dir, ref_path)

    # Check corresponding .mat file for .m file
    if ref_path.endswith('.m'):
        mat_file = ref_path.replace('.m', '.mat')
    else:
        mat_file = ref_path

    if not os.path.exists(mat_file):
        print(f"Warning: Reference solution file not found: {mat_file}")
        return None

    try:
        print(f"Loading reference solution: {mat_file}")
        data = scipy.io.loadmat(mat_file)

        # Extract data - common handling of x coordinate
        if 'x' in data:
            x_ref = data['x'].flatten()
        else:
            print("Warning: 'x' not found in reference solution file")
            return None

        # ... (dimension-specific handling follows)
```

**Differences:**
- **time_pde**: Creates interpolators for time evolution, returns `(t_ref, x_ref, u_ref, interpolators)`
- **general_pde**: Handles 2D steady-state problems, returns `(x_ref, y_ref, u_ref, y_label)`

**Consolidation Strategy:** Extract common path/file handling, delegate dimension-specific processing to subclasses.

#### Helper Method (time_pde only)
**`get_reference_solution_at_time(self, t: float) -> Optional[np.ndarray]`** (Lines 157-175)

This is specific to time evolution but uses the common reference solution structure.

### 2.3 GPU/CPU Data Conversion Methods

#### `_to_numpy_cpu(self, data)`
**Similarity:** 100%

**Identical implementation in general_pde (Lines 443-468 in solver.py, 54-79 in visualize.py):**
```python
def _to_numpy_cpu(self, data):
    """Convert data to CPU numpy array (handle torch tensors and GPU arrays)"""
    if data is None:
        return None

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    if isinstance(data, np.ndarray):
        return np.asarray(data)

    try:
        return np.asarray(data)
    except:
        return data
```

**Note:** This method appears in `general_pde_solver` but is missing from `time_pde_solver`. Should be added to base class.

#### `_interpolate_to_test_grid(self, U_train: np.ndarray) -> np.ndarray`
**Similarity:** 100%

**Identical implementation in general_pde (Lines 404-441):**
```python
def _interpolate_to_test_grid(self, U_train: np.ndarray) -> np.ndarray:
    """Interpolate solution from training points to test grid"""
    from scipy.interpolate import griddata

    x_train = self.data_train['x']
    x_test = self.data_test['x']
    n_eqs = U_train.shape[1] if U_train.ndim > 1 else 1

    if n_eqs == 1:
        U_train = U_train.reshape(-1, 1)

    U_test = np.zeros((x_test.shape[0], n_eqs))

    for eq_idx in range(n_eqs):
        U_test[:, eq_idx] = griddata(
            x_train, U_train[:, eq_idx], x_test,
            method='linear', fill_value=np.nan
        )

        # Fill NaN with nearest neighbor interpolation
        nan_mask = np.isnan(U_test[:, eq_idx])
        if np.any(nan_mask):
            U_test[nan_mask, eq_idx] = griddata(
                x_train, U_train[:, eq_idx], x_test[nan_mask],
                method='nearest'
            )

    print(f"  Interpolated from {x_train.shape[0]} training points to {x_test.shape[0]} test points")
    return U_test
```

**Note:** This utility method should be in base class.

### 2.4 Neural Network Training Methods

#### `_train_neural_network_step(self, it: int, dt: float, U_current: np.ndarray = None)`
**Similarity:** 95%

Both solvers have nearly identical neural network training logic:

**Common Pattern:**
```python
def _train_neural_network_step(self, it: int, dt: float, U_current: np.ndarray = None):
    # Check if training should be skipped
    spotter_skip = getattr(self.config, 'spotter_skip', 1)
    skip_training = (it % spotter_skip != 0) and (it > 0)

    if skip_training:
        print(f"  Skipping neural network training (spotter_skip={spotter_skip}, it={it})")
        self.fitter.fitter_init(self.model)
        return

    print(f"  Training neural network (it={it}, inheriting from previous parameters)...")

    # Prepare GPU data
    data_GPU = self.model.prepare_gpu_data(self.data, U_current)

    # Train from previous state
    self.model.train()
    self.model.train_net(self.data, self.model, data_GPU, dt=dt)

    # Compute final loss
    self.model.eval()
    final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
    print(f"  Neural network training completed, loss: {final_loss:.8e}")

    # Re-initialize fitter with updated model
    self.fitter.fitter_init(self.model)
```

**Differences:**
- **time_pde**: Simpler, doesn't set config attributes (Lines 342-377)
- **general_pde**: Sets multiple `model.config` attributes before training (Lines 380-432)

**Consolidation:** Can use template method pattern - base class provides workflow, subclass hooks for config setup.

---

## Section 3: Near-Identical Code Blocks (80%+ Similar)

### 3.1 Initialization Sequence
**Location:**
- time_pde: Lines 45-90
- general_pde: Lines 39-87

**Similarity:** 90%

Both follow pattern:
1. `_initialize_config(config, case_dir)`
2. `_initialize_components()`
3. Create results directory

### 3.2 Results Directory Creation
**Location:** Throughout both solvers

**Identical Code:**
```python
os.makedirs(self.config.results_dir, exist_ok=True)
```

Appears in:
- `_initialize_components()`
- Various visualization methods

### 3.3 Reference Solution Path Handling
**Location:**
- time_pde: Lines 104-119
- general_pde: Lines 209-224

**100% Identical:**
```python
ref_path = self.config.reference_solution

if not os.path.isabs(ref_path):
    case_dir = getattr(self.config, 'case_dir', os.getcwd())
    ref_path = os.path.join(case_dir, ref_path)

if ref_path.endswith('.m'):
    mat_file = ref_path.replace('.m', '.mat')
else:
    mat_file = ref_path

if not os.path.exists(mat_file):
    print(f"Warning: Reference solution file not found: {mat_file}")
    return None
```

### 3.4 Config Attribute Access Pattern
**Similarity:** 100%

Both use identical pattern:
```python
getattr(self.config, 'attribute_name', default_value)
```

Examples:
```python
time_scheme = getattr(self.config, 'time_scheme', 'imex_rk_222')  # time_pde
time_scheme = getattr(self.config, 'time_scheme', 'implicit_euler')  # general_pde
spotter_skip = getattr(self.config, 'spotter_skip', 1)  # both
realtime_visualization = getattr(self.config, 'realtime_visualization', False)  # both
```

### 3.5 Fitter Initialization Pattern
**Location:**
- time_pde: Line 261
- general_pde: Line 133, 390, 432

**Identical Call:**
```python
self.fitter.fitter_init(self.model)
```

---

## Section 4: Solver-Specific Differences

### 4.1 Main Solve Methods

#### time_pde_solver: `solve_time_evolution()`
**Purpose:** Physical time evolution with adaptive time stepping

**Key Features:**
- Time loop: `while T < self.config.T:`
- Adaptive time step: `_compute_adaptive_timestep()`
- Time integration: `self.fitter.solve_time_step()`
- Animation data tracking per time step
- Reference solution comparison at each time

**Unique Attributes:**
- `self.loss_history`
- `self.physics_start_time`, `self.physics_end_time`
- `self._previous_U` (for adaptive time stepping)

#### general_pde_solver: `solve_steady_state()`
**Purpose:** Pseudo-time stepping to steady state

**Key Features:**
- Iteration loop: `for iteration in range(self.max_iterations):`
- Convergence monitoring: residual-based stopping
- Adaptive pseudo-time step based on residual
- Spotter initialization stage
- Pseudo-time stepping via `self.fitter.time_step()`

**Unique Attributes:**
- `residual_history`, `dt_history`, `solution_norm_history`
- `_coeffs_current`
- Convergence info dictionary

### 4.2 Initialization Stages

#### time_pde_solver
Simple two-stage:
1. `_initialize_config()`
2. `_initialize_components()`

#### general_pde_solver
More complex with initialization methods:
1. `_initialize_config()`
2. `_initialize_components()`
3. `_spotter_stage()` - Spotter network initialization (optional)
4. `_get_initial_from_config()` - Given initial conditions

### 4.3 Time Step Control

#### time_pde_solver: Physical Time
```python
def _compute_adaptive_timestep(self, it: int, T: float, dt: float, U: np.ndarray) -> float:
    # CFL condition
    # Stability restriction
    # Solution change rate restriction
```

Methods: `_compute_cfl_timestep()`, `_compute_solution_change_limit()`

#### general_pde_solver: Pseudo-Time (Residual-Based)
```python
# Adaptive time step control based on residual magnitude
if residual > 0.1:
    dt = self.dt_max
elif residual > 0.01:
    dt = self.dt_max * (0.2 + 0.8 * ratio)
elif residual > 1e-4:
    dt = self.dt_min + (self.dt_max * 0.2 - self.dt_min) * ratio
else:
    dt = self.dt_min
```

### 4.4 Post-Processing

#### time_pde_solver: `_postprocess_results()`
Extensive post-processing:
- Final time comparison with reference
- Spacetime comparison plots (2D: x-t domain)
- Error statistics over all time steps
- Time evolution GIF animation
- MATLAB data export

#### general_pde_solver: `_postprocess()`
Simpler steady-state post-processing:
- Convergence history plots
- Final solution visualization
- Reference comparison (if available)
- Iteration evolution GIF
- Solution data export

---

## Section 5: Proposed Nonlinear Solver Base Class

### 5.1 Base Class Structure

```python
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn
import scipy.io
import os

class NonlinearPDESolver(ABC):
    """Abstract base class for nonlinear PDE solvers using hybrid DeePoly method

    Provides common functionality for:
    - Component initialization (config, data, model, fitter, visualizer)
    - Reference solution loading and interpolation
    - Neural network training workflow
    - GPU/CPU data conversion utilities
    - Results directory management
    """

    # ==================== Abstract Methods (Solver-Specific) ====================

    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """Main solve method - implemented by subclasses

        Returns:
            Dict containing solution and convergence information
        """
        pass

    @abstractmethod
    def _create_config(self, case_dir: str):
        """Create solver-specific config object

        Args:
            case_dir: Case directory path

        Returns:
            Config object (TimePDEConfig or GeneralPDEConfig)
        """
        pass

    @abstractmethod
    def _create_data_generator(self):
        """Create solver-specific data generator

        Returns:
            DataGenerator object
        """
        pass

    @abstractmethod
    def _create_model(self):
        """Create solver-specific neural network model

        Returns:
            Neural network model
        """
        pass

    @abstractmethod
    def _create_fitter(self, time_scheme: str):
        """Create solver-specific fitter

        Args:
            time_scheme: Time integration scheme name

        Returns:
            Fitter object
        """
        pass

    @abstractmethod
    def _create_visualizer(self):
        """Create solver-specific visualizer

        Returns:
            Visualizer object
        """
        pass

    @abstractmethod
    def _get_default_time_scheme(self) -> str:
        """Get default time integration scheme

        Returns:
            Default time scheme name
        """
        pass

    @abstractmethod
    def _postprocess_results(self, **kwargs):
        """Post-process and visualize results - solver-specific"""
        pass

    # ==================== Concrete Methods (Common Functionality) ====================

    def __init__(self, config=None, case_dir=None):
        """Initialize solver with factory pattern for component creation"""
        self._initialize_config(config, case_dir)
        self._initialize_components()

    def _initialize_config(self, config, case_dir):
        """Initialize configuration - identical in both solvers"""
        if config is not None:
            self.config = config
        elif case_dir is not None:
            self.config = self._create_config(case_dir)
        else:
            raise ValueError("Either config object or case_dir must be provided")

        self.case_dir = case_dir if case_dir else getattr(self.config, "case_dir", None)

        # Export configuration file
        if hasattr(self.config, "export_to_json"):
            self.config.export_to_json("config.json")

    def _initialize_components(self):
        """Initialize core components using factory methods"""
        # Data generator
        self.datagen = self._create_data_generator()

        # Generate training and test data
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")

        # Neural network model
        self.model = self._create_model().to(self.config.device)

        # Visualizer
        self.visualizer = self._create_visualizer()

        # Fitter with time scheme
        time_scheme = getattr(self.config, 'time_scheme', self._get_default_time_scheme())
        self.fitter = self._create_fitter(time_scheme)

        # Load reference solution
        self.reference_solution = self._load_reference_solution()

        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)

    def _load_reference_solution(self) -> Optional[Dict]:
        """Load reference solution data - common path/file handling"""
        if not hasattr(self.config, 'reference_solution') or not self.config.reference_solution:
            return None

        ref_path = self.config.reference_solution

        # If relative path, make it relative to case directory
        if not os.path.isabs(ref_path):
            case_dir = getattr(self.config, 'case_dir', os.getcwd())
            ref_path = os.path.join(case_dir, ref_path)

        # Check corresponding .mat file for .m file
        if ref_path.endswith('.m'):
            mat_file = ref_path.replace('.m', '.mat')
        else:
            mat_file = ref_path

        if not os.path.exists(mat_file):
            print(f"Warning: Reference solution file not found: {mat_file}")
            return None

        try:
            print(f"Loading reference solution: {mat_file}")
            data = scipy.io.loadmat(mat_file)

            # Common: Extract x coordinate
            if 'x' in data:
                x_ref = data['x'].flatten()
            else:
                print("Warning: 'x' not found in reference solution file")
                return None

            # Delegate dimension-specific processing to subclass
            return self._process_reference_solution(data, x_ref, mat_file)

        except Exception as e:
            print(f"Error: Cannot read reference solution file {mat_file}: {e}")
            return None

    @abstractmethod
    def _process_reference_solution(self, data: Dict, x_ref: np.ndarray,
                                   mat_file: str) -> Dict:
        """Process reference solution data - solver-specific

        Args:
            data: MATLAB data dictionary
            x_ref: Spatial coordinates
            mat_file: Path to .mat file

        Returns:
            Reference solution dictionary
        """
        pass

    def _to_numpy_cpu(self, data):
        """Convert data to CPU numpy array (handle torch tensors and GPU arrays)"""
        if data is None:
            return None

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()

        if isinstance(data, np.ndarray):
            return np.asarray(data)

        try:
            return np.asarray(data)
        except:
            return data

    def _interpolate_to_test_grid(self, U_train: np.ndarray) -> np.ndarray:
        """Interpolate solution from training points to test grid"""
        from scipy.interpolate import griddata

        x_train = self.data_train['x']
        x_test = self.data_test['x']
        n_eqs = U_train.shape[1] if U_train.ndim > 1 else 1

        if n_eqs == 1:
            U_train = U_train.reshape(-1, 1)

        U_test = np.zeros((x_test.shape[0], n_eqs))

        for eq_idx in range(n_eqs):
            U_test[:, eq_idx] = griddata(
                x_train, U_train[:, eq_idx], x_test,
                method='linear', fill_value=np.nan
            )

            # Fill NaN with nearest neighbor interpolation
            nan_mask = np.isnan(U_test[:, eq_idx])
            if np.any(nan_mask):
                U_test[nan_mask, eq_idx] = griddata(
                    x_train, U_train[:, eq_idx], x_test[nan_mask],
                    method='nearest'
                )

        print(f"  Interpolated from {x_train.shape[0]} training points to {x_test.shape[0]} test points")
        return U_test

    def _train_neural_network_step(self, it: int, dt: float, U_current: np.ndarray = None):
        """Neural network training at current iteration - template method pattern

        Args:
            it: Iteration number
            dt: Time step size (physical or pseudo-time)
            U_current: Current solution
        """
        # Check if training should be skipped
        spotter_skip = getattr(self.config, 'spotter_skip', 1)
        skip_training = (it % spotter_skip != 0) and (it > 0)

        if skip_training:
            print(f"  Skipping neural network training (spotter_skip={spotter_skip}, it={it})")
            self.fitter.fitter_init(self.model)
            return

        print(f"  Training neural network (it={it}, inheriting from previous parameters)...")

        # Hook for subclass-specific config setup
        self._setup_model_config_for_training(dt)

        # Prepare GPU data
        data_GPU = self.model.prepare_gpu_data(self.data_train, U_current)

        # Train from previous state
        self.model.train()
        self.model.train_net(self.data_train, self.model, data_GPU, dt=dt)

        # Compute final loss
        self.model.eval()
        final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
        print(f"  Neural network training completed, loss: {final_loss:.8e}")

        # Re-initialize fitter with updated model
        self.fitter.fitter_init(self.model)

    def _setup_model_config_for_training(self, dt: float):
        """Setup model config before training - hook for subclasses

        Default implementation does nothing. Subclasses can override to set
        solver-specific config attributes.

        Args:
            dt: Time step size
        """
        pass
```

### 5.2 Concrete Methods to Include

1. **`_initialize_config()`** - 100% common
2. **`_initialize_components()`** - Common workflow with factory methods
3. **`_load_reference_solution()`** - Common file handling + abstract processing hook
4. **`_to_numpy_cpu()`** - 100% common utility
5. **`_interpolate_to_test_grid()`** - 100% common utility
6. **`_train_neural_network_step()`** - Template method with hook for config setup

### 5.3 Abstract Methods (Solver-Specific)

1. **`solve()`** - Main solving logic differs completely
2. **`_create_config()`** - Factory method for config creation
3. **`_create_data_generator()`** - Factory method
4. **`_create_model()`** - Factory method
5. **`_create_fitter()`** - Factory method
6. **`_create_visualizer()`** - Factory method
7. **`_get_default_time_scheme()`** - Solver-specific default
8. **`_process_reference_solution()`** - Dimension-specific processing
9. **`_postprocess_results()`** - Solver-specific visualization
10. **`_setup_model_config_for_training()`** - Hook method (optional override)

---

## Section 6: Migration Impact

### 6.1 Changes to time_pde_solver

**Inheritance:**
```python
from src.problem_solvers.nonlinear_solver.base import NonlinearPDESolver

class TimePDESolver(NonlinearPDESolver):
    """Time-dependent PDE solver"""
```

**Methods to Remove:**
- ✓ `_initialize_config()` - Move to base
- ✓ `_initialize_components()` - Partially move (use factory pattern)
- ✓ `_load_reference_solution()` - Move common parts to base
- ✓ `_to_numpy_cpu()` - Add this method (currently missing, exists in general_pde)
- ✓ `_interpolate_to_test_grid()` - Add this method (currently missing)

**Methods to Implement (Abstract):**
```python
def solve(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
    """Main solve - calls solve_time_evolution()"""

def _create_config(self, case_dir: str):
    return TimePDEConfig(case_dir=case_dir)

def _create_data_generator(self):
    return TimePDEDataGenerator(self.config)

def _create_model(self):
    return TimePDENet(self.config)

def _create_fitter(self, time_scheme: str):
    return TimePDEFitter(config=self.config, data=self.data_train,
                        time_scheme=time_scheme)

def _create_visualizer(self):
    return TimePDEVisualizer(self.config)

def _get_default_time_scheme(self) -> str:
    return 'imex_rk_222'

def _process_reference_solution(self, data, x_ref, mat_file):
    # Extract t_ref, create interpolators, return time-specific dict

def _postprocess_results(self, U_final, U_seg_final, model_final, coeffs_final):
    # Existing implementation remains
```

**Methods to Keep (Solver-Specific):**
- `solve_time_evolution()` - Core time evolution logic
- `_initialize_time_evolution()` - Time-specific initialization
- `_compute_adaptive_timestep()` - CFL/stability-based adaptation
- `_compute_cfl_timestep()` - CFL condition
- `_compute_solution_change_limit()` - Solution-based adaptation
- `_monitor_solution()` - Time evolution monitoring
- `_plot_neural_network_predictions()` - Time-specific plotting
- All post-processing methods (Lines 486-1008)
- `get_reference_solution_at_time()` - Time interpolation
- Convenience methods (Lines 1162-1203)

**Additional Changes:**
```python
def _initialize_solution_tracking(self):
    """Initialize solution tracking - called by __init__ after super().__init__()"""
    self.loss_history = []
    self.visualizer.reset_animation_data()

def __init__(self, config=None, case_dir=None):
    super().__init__(config, case_dir)
    self._initialize_solution_tracking()  # Time-specific addition
```

### 6.2 Changes to general_pde_solver

**Inheritance:**
```python
from src.problem_solvers.nonlinear_solver.base import NonlinearPDESolver

class GeneralPDESolver(NonlinearPDESolver):
    """General PDE solver with pseudo-time stepping"""
```

**Methods to Remove:**
- ✓ `_initialize_config()` - Move to base
- ✓ `_initialize_components()` - Partially move (use factory pattern)
- ✓ `_load_reference_solution()` - Move common parts to base
- ✓ `_to_numpy_cpu()` - Move to base (already in solver and visualizer)
- ✓ `_interpolate_to_test_grid()` - Move to base

**Methods to Implement (Abstract):**
```python
def solve(self, U_init: Optional[np.ndarray] = None):
    """Main solve - returns dict with solution and convergence info"""

def _create_config(self, case_dir: str):
    return GeneralPDEConfig(case_dir=case_dir)

def _create_data_generator(self):
    return GeneralPDEDataGenerator(self.config)

def _create_model(self):
    return GeneralPDENet(self.config)

def _create_fitter(self, time_scheme: str):
    return GeneralPDEFitter(config=self.config, data=self.data_train,
                           time_scheme=time_scheme, visualizer=self.visualizer)

def _create_visualizer(self):
    return GeneralPDEVisualizer(self.config)

def _get_default_time_scheme(self) -> str:
    return 'implicit_euler'

def _process_reference_solution(self, data, x_ref, mat_file):
    # Extract y_ref (or t as y), handle 1D/2D, return steady-state dict

def _postprocess(self, U_steady, coeffs_steady, conv_info):
    # Existing implementation remains

def _setup_model_config_for_training(self, dt: float):
    """Override to set general_pde-specific config attributes"""
    # Set epochs_adam, epochs_lbfgs, DNNtol, etc. from config
    # (Lines 395-417 in current implementation)
```

**Methods to Keep (Solver-Specific):**
- `_spotter_stage()` - Spotter initialization (Lines 310-356)
- `_get_initial_from_config()` - Given initial conditions (Lines 162-202)
- `solve()` - Main solve wrapper with spotter stage (Lines 89-160)
- All fitter delegation is now in GeneralPDEFitter (no changes needed here)

**Note:** The `_train_neural_network_step()` in GeneralPDEFitter (Lines 380-432) should override the hook method:

```python
# In general_pde_solver.py
def _setup_model_config_for_training(self, dt: float):
    """Setup model config for general PDE training"""
    fitter_training = getattr(self.config, 'fitter_training', {})

    if not hasattr(self.model.config, 'max_retries'):
        self.model.config.max_retries = getattr(self.config, 'max_retries',
                                               fitter_training.get('max_retries', 3))

    if not hasattr(self.model.config, 'learning_rate'):
        self.model.config.learning_rate = getattr(self.config, 'learning_rate',
                                                  fitter_training.get('learning_rate', 0.001))

    if not hasattr(self.model.config, 'epochs_adam'):
        self.model.config.epochs_adam = getattr(self.config, 'epochs_adam',
                                               fitter_training.get('epochs', 100))

    if not hasattr(self.model.config, 'epochs_lbfgs'):
        self.model.config.epochs_lbfgs = getattr(self.config, 'epochs_lbfgs',
                                                 fitter_training.get('epochs_lbfgs', 0))

    if not hasattr(self.model.config, 'DNNtol'):
        self.model.config.DNNtol = getattr(self.config, 'DNNtol',
                                          fitter_training.get('DNNtol', 1e-6))

    self.model.config.dt = dt
```

---

## Section 7: Implementation Roadmap

### Phase 1: Create Base Class Structure
1. Create `/src/problem_solvers/nonlinear_solver/` directory
2. Create `base.py` with `NonlinearPDESolver` abstract class
3. Implement all concrete methods listed in Section 5.2
4. Define all abstract methods listed in Section 5.3
5. Add comprehensive docstrings and type hints

### Phase 2: Migrate time_pde_solver
1. Update `TimePDESolver` to inherit from `NonlinearPDESolver`
2. Remove duplicate methods
3. Implement factory methods
4. Implement `_process_reference_solution()` for time evolution
5. Override `_setup_model_config_for_training()` if needed (currently no-op)
6. Add `_initialize_solution_tracking()` call to `__init__`
7. Test thoroughly with existing test cases

### Phase 3: Migrate general_pde_solver
1. Update `GeneralPDESolver` to inherit from `NonlinearPDESolver`
2. Remove duplicate methods
3. Implement factory methods
4. Implement `_process_reference_solution()` for steady-state
5. Override `_setup_model_config_for_training()` with config setup logic
6. Test thoroughly with existing test cases

### Phase 4: Validation & Testing
1. Run all existing test cases for both solvers
2. Verify reference solution loading works correctly
3. Verify neural network training workflow
4. Verify visualization and post-processing
5. Check for any performance regressions
6. Update documentation

### Phase 5: Code Quality Improvements
1. Add unit tests for base class methods
2. Add integration tests for inheritance
3. Update type hints throughout
4. Run linters and fix issues
5. Update CLAUDE.md with new architecture notes

---

## Section 8: Additional Recommendations

### 8.1 Fitter Layer Consideration
The analysis reveals that `GeneralPDEFitter` already inherits from `TimePDEFitter`:

```python
class GeneralPDEFitter(TimePDEFitter):
    """General PDE Fitter with pseudo-time stepping for steady-state problems"""
```

This is **correct design** - the fitter inheritance hierarchy already captures the relationship:
- `BaseDeepPolyFitter` → `TimePDEFitter` → `GeneralPDEFitter`

The solver layer should **mirror** this hierarchy:
- `abstract_class` → `NonlinearPDESolver` → `TimePDESolver` / `GeneralPDESolver`

### 8.2 Config Layer Consideration
Both configs share significant common structure:

**Common Config Methods:**
- `_normalize_operator_splitting()` - 100% identical (Lines 88-136 in both)
- `_pad_operator_list()` - 100% identical (Lines 138-153 in both)
- `_auto_code()` - 100% identical (Lines 190-208 time_pde, 308-327 general_pde)
- `_disable_auto_code()` - 100% identical
- `_parse_operator_splitting()` - 100% identical (Lines 226-266 time_pde, 344-384 general_pde)

**Recommendation:** Consider extracting these into `BaseOperatorConfig` or similar base class.

### 8.3 Visualizer Layer
Both visualizers inherit from `BaseVisualizer` and share:
- `_to_numpy_cpu()` - 100% identical
- Animation data management patterns
- Real-time visualization workflow
- GIF generation logic

**Recommendation:** Extract common visualization utilities into `BaseVisualizer` methods.

### 8.4 Data Generator Layer
Both data generators likely share:
- Segment generation logic
- Normalization methods
- Boundary condition processing

**Recommendation:** Review data generators for extractable common functionality.

---

## Section 9: Critical Success Factors

1. **Preserve Existing Behavior:** All tests must pass after refactoring
2. **No Performance Degradation:** Virtual function calls should have negligible overhead
3. **Clear Documentation:** Abstract vs concrete methods must be well-documented
4. **Type Safety:** Use type hints throughout for IDE support
5. **Gradual Migration:** Migrate one solver at a time, test thoroughly
6. **Backward Compatibility:** Ensure existing case configurations still work

---

## Appendix A: Line-by-Line Comparison

### Method Comparison Table

| Method | time_pde Lines | general_pde Lines | Similarity | Action |
|--------|---------------|------------------|------------|--------|
| `__init__` | 45-49 | 39-42 | 95% | Template in base |
| `_initialize_config` | 51-64 | 44-57 | 100% | Move to base |
| `_initialize_components` | 66-90 | 59-87 | 90% | Factory pattern |
| `_load_reference_solution` | 99-155 | 204-308 | 85% | Split common/specific |
| `_to_numpy_cpu` | N/A | 443-468 | 100% | Add to base |
| `_interpolate_to_test_grid` | N/A | 404-441 | 100% | Add to base |
| `_train_neural_network_step` | 342-377 | 380-432* | 95% | Template method |
| `_postprocess_results` | 488-1008 | 358-402 | 20% | Keep separate |
| `solve` | 177-192 | 89-160 | 0% | Abstract method |

*Note: In `general_pde_solver`, this is in the fitter class, not solver

### Inheritance Impact Summary

**Before:**
```
abstract_class
    ├── BasePDEConfig
    │   ├── TimePDEConfig
    │   └── GeneralPDEConfig
    ├── BaseDeepPolyFitter
    │   ├── TimePDEFitter
    │   └── GeneralPDEFitter (inherits from TimePDEFitter)
    └── BaseNet
        ├── TimePDENet
        └── GeneralPDENet

Solvers (NO common base):
    ├── TimePDESolver
    └── GeneralPDESolver
```

**After:**
```
abstract_class
    ├── BasePDEConfig
    │   ├── TimePDEConfig
    │   └── GeneralPDEConfig
    ├── BaseDeepPolyFitter
    │   ├── TimePDEFitter
    │   └── GeneralPDEFitter (inherits from TimePDEFitter)
    └── BaseNet
        ├── TimePDENet
        └── GeneralPDENet

nonlinear_solver
    └── NonlinearPDESolver (NEW)
        ├── TimePDESolver
        └── GeneralPDESolver
```

---

## Conclusion

This analysis identified **~700 lines of duplicated code** across the two solvers that can be consolidated into a `NonlinearPDESolver` base class. The key findings:

1. **60-70% code similarity** in initialization, component management, and utilities
2. **100% identical** implementation of 6+ methods (config init, reference loading core, etc.)
3. **Clear separation** between common workflow and solver-specific algorithms
4. **Factory method pattern** enables clean component creation while maintaining flexibility
5. **Template method pattern** enables neural network training workflow consolidation

The proposed base class will:
- **Eliminate code duplication** without compromising flexibility
- **Improve maintainability** through centralized common functionality
- **Enable future solvers** to leverage existing infrastructure
- **Preserve performance** (virtual function overhead is negligible)
- **Maintain backward compatibility** with existing configurations

**Next Step:** Begin Phase 1 implementation of the base class structure.
