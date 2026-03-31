"""Pseudo-time fitter facade with split runtime, residual, and Jacobian helpers."""

from typing import Dict

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver
from .pseudotime_fitter_jacobian_mixin import PseudoTimeFitterJacobianMixin
from .pseudotime_fitter_residual_mixin import PseudoTimeFitterResidualMixin
from .pseudotime_fitter_runtime_mixin import PseudoTimeFitterRuntimeMixin


class PseudoTimePDEFitter(
    PseudoTimeFitterRuntimeMixin,
    PseudoTimeFitterResidualMixin,
    PseudoTimeFitterJacobianMixin,
    BaseDeepPolyFitter,
):
    """Pseudo-Time PDE Fitter with pseudo-time stepping for steady-state problems

    Features:
    - Pseudo-time stepping iteration
    - Automatic convergence monitoring
    - Adaptive pseudo-time step control
    - Residual-based stopping criteria
    """

    # Adaptive time-stepping thresholds
    ADAPTIVE_DT_THRESHOLD_HIGH = 0.1      # Above this: use dt_max
    ADAPTIVE_DT_THRESHOLD_MID = 0.01      # Above this: interpolate between 0.2*dt_max and dt_max
    ADAPTIVE_DT_THRESHOLD_LOW = 1e-4      # Above this: interpolate between dt_min and 0.2*dt_max
    ADAPTIVE_DT_SCALE_FACTOR = 0.2        # Scale factor for mid-range dt
    ADAPTIVE_DT_INTERP_HIGH = 0.8         # Interpolation weight for high range

    # Convergence checking
    CONVERGENCE_CHECK_FACTOR = 10         # Check convergence more frequently when residual < tol * this factor

    # Initialization
    COEFF_PERTURBATION = 1e-8             # Random perturbation to break symmetry

    # Neural network training defaults
    NN_DEFAULT_MAX_RETRIES = 3
    NN_DEFAULT_LEARNING_RATE = 0.001
    NN_DEFAULT_EPOCHS_ADAM = 100
    NN_DEFAULT_EPOCHS_LBFGS = 0
    NN_DEFAULT_TOL = 1e-6

    # Error handling
    MAX_CONSECUTIVE_FAILURES = 5          # Max consecutive failures before giving up

    def __init__(self, config, data: Dict = None, visualizer=None):
        super().__init__(config, data=data)
        self.solver = LinearSolver(
            verbose=True,
            use_gpu=True,
            performance_tracking=True,
            config=config,
        )
        self.max_iterations = config.max_iterations
        self.residual_tol = config.residual_tol
        self.dt_init = config.dt_init
        self.dt_min = config.dt_min
        self.dt_max = config.dt_max
        self.adaptive_dt = config.adaptive_dt
        self.dt_increase_factor = config.dt_increase_factor
        self.dt_decrease_factor = config.dt_decrease_factor
        self.check_interval = config.check_interval
        self.print_interval = config.print_interval
        self.save_interval = config.save_interval
        self.max_residual = config.max_residual
        self.residual_stagnation_steps = config.residual_stagnation_steps
        self.convergence_primary_var = getattr(config, "convergence_primary_var", 0)
        self.convergence_primary_tol = getattr(config, "convergence_primary_tol", 1e-4)
        self.convergence_improvement_factor = getattr(config, "convergence_improvement_factor", 0.999)
        self.convergence_plateau_steps = getattr(config, "convergence_plateau_steps", 10)
        self.convergence_plateau_tol = getattr(config, "convergence_plateau_tol", 1e-2)
        self.convergence_dt_growth_tol = getattr(config, "convergence_dt_growth_tol", 1e-3)
        self.rollback_on_plateau = getattr(config, "rollback_on_plateau", True)
        self.rollback_dt_min = getattr(config, "rollback_dt_min", self.dt_min)
        self.force_retrain_on_plateau = getattr(config, "force_retrain_on_plateau", True)
        self.return_best_on_exit = getattr(config, "return_best_on_exit", True)
        self.residual_history = []
        self.dt_history = []
        self.solution_norm_history = []
        self.primary_update_history = []
        self.primary_relative_update_history = []
        self.max_abs_update_history = []
        self.best_primary_update = float("inf")
        self.best_iteration = -1
        self.restart_count = 0
        self.restart_history = []
        self._coeffs_current = None
        self.visualizer = visualizer
        self._segment_start_indices = None
        self._segment_end_indices = None
