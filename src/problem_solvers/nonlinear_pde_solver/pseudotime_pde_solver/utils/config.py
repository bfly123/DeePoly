"""Configuration for general PDE solver with pseudo-time stepping."""

from dataclasses import dataclass, field
from typing import Dict, List

from src.abstract_class.config.base_pde_config import BasePDEConfig
from .pseudotime_config_loading_mixin import PseudotimeConfigLoadingMixin
from .pseudotime_config_misc_mixin import PseudotimeConfigMiscMixin
from .pseudotime_config_operator_mixin import PseudotimeConfigOperatorMixin


@dataclass
class PseudoTimePDEConfig(
    PseudotimeConfigMiscMixin,
    PseudotimeConfigLoadingMixin,
    PseudotimeConfigOperatorMixin,
    BasePDEConfig,
):
    """Configuration class for general PDE problems with pseudo-time stepping

    Extends BasePDEConfig with general PDE specific features:
    - Operator splitting (L1, L2, F, N) with forced existence (same as TimePDE)
    - Pseudo-time stepping parameters (instead of physical time T and dt)
    - Initial condition handling (Spotter training or given)
    - Convergence monitoring and adaptive time step control
    """

    # Pseudo-time stepping parameters (replace T and dt from TimePDE)
    max_iterations: int = 1000
    residual_tol: float = 1e-6
    dt_init: float = 0.1
    dt_min: float = 1e-5
    dt_max: float = 0.5
    adaptive_dt: bool = True
    dt_increase_factor: float = 1.2
    dt_decrease_factor: float = 0.5
    check_interval: int = 10
    print_interval: int = 50
    save_interval: int = 100
    max_residual: float = 1e10
    residual_stagnation_steps: int = 50
    convergence_primary_var: int = 0
    convergence_primary_tol: float = 1e-4
    convergence_improvement_factor: float = 0.999
    convergence_plateau_steps: int = 10
    convergence_plateau_tol: float = 1e-2
    convergence_dt_growth_tol: float = 1e-3
    rollback_on_plateau: bool = True
    rollback_dt_min: float = 1e-2
    force_retrain_on_plateau: bool = True
    return_best_on_exit: bool = True
    reference_stable_single_retrain: bool = True
    reference_stable_retrain_stop_tol: float = 0.0
    reference_stable_adaptive_tail: bool = False

    # Snipper solving mode control
    steady_state_mode: bool = True  # Pseudo-time solver defaults to steady-state mode
    pseudo_time_formulation: str = "default"  # default | reference_stable
    pseudo_time_direction: float = -1.0
    nonlinear_linearization: str = "picard"  # picard | newton
    mass_eq_mask: List[int] = field(default_factory=list)
    mass_var_map: List[int] = field(default_factory=list)
    frozen_nonlinear_jacobian: str = "none"  # none | transport

    # Legacy training parameter aliases kept for runtime compatibility.
    epochs_adam: int = 5000
    epochs_lbfgs: int = 0
    max_retries: int = 1
    DNNtol: float = 1e-4
    spotter_skip: int = 1  # Neural network training frequency (1 = every iteration)

    # Preferred pseudo-time network-training configuration.
    spotter: Dict = field(default_factory=dict)

    # Initial condition configuration (Spotter or given)
    initial_condition: Dict = field(default_factory=dict)

    # Legacy fitter-training block. Pseudo-time cases should prefer `spotter`.
    fitter_training: Dict = field(default_factory=dict)

    # Internal implementation details kept for backward compatibility.
    time_scheme: str = "implicit_euler"
    time_scheme_params: Dict = field(default_factory=dict)

    # Operator splitting configuration (same as TimePDE)
    eq_L1: List[str] = field(default_factory=list)
    eq_L2: List[str] = field(default_factory=list)
    eq_F: List[str] = field(default_factory=list)
    eq_N: List[str] = field(default_factory=list)

    # Legacy backward compatibility fields
    eq_linear_list: List[str] = field(default_factory=list)
    deriv_orders: List[int] = field(default_factory=list)
    max_deriv_orders: List[int] = field(default_factory=list)
    eq_nonlinear_list: List[str] = field(default_factory=list)
    all_derivatives: List[List[int]] = field(default_factory=list)

    # Backward compatibility aliases
    f_L2: List[str] = field(default_factory=list, init=False)
    N: List[str] = field(default_factory=list, init=False)

    def _solver_specific_init(self):
        """Pseudo-Time PDE specific initialization"""
        # Extract operator splitting from eq field (same as TimePDE)
        self._normalize_operator_splitting()

        # Load pseudo-time stepping parameters
        self._load_pseudo_time_config()

        # Load initial condition config
        self._load_initial_condition_config()

        # Load pseudo-time network-training config
        self._load_spotter_config()

        # Load visualization config
        self._load_visualization_config()

        # Set results directory
        self.results_dir = self.get_results_dir()

        # Perform auto-code generation if enabled
        self._auto_code()

        # Parse equations with operator splitting (same as TimePDE)
        self._parse_operator_splitting()

    def _calculate_n_eqs(self):
        """Calculate number of equations for general PDE

        For general PDEs, n_eqs equals the number of variables since
        L1, L2, F, N are different operator terms of the same PDE.
        """
        return len(self.vars_list)
