"""Configuration for general PDE solver with pseudo-time stepping."""

from dataclasses import dataclass, field
from typing import Dict, List

from src.abstract_class.config.base_pde_config import BasePDEConfig
from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.fixiteration_config_loading_mixin import (
    FixIterationConfigLoadingMixin,
)
from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.fixiteration_config_misc_mixin import (
    FixIterationConfigMiscMixin,
)
from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.fixiteration_config_operator_mixin import (
    FixIterationConfigOperatorMixin,
)


@dataclass
class FixIterationPDEConfig(
    FixIterationConfigOperatorMixin,
    FixIterationConfigLoadingMixin,
    FixIterationConfigMiscMixin,
    BasePDEConfig,
):
    max_iterations: int = 1000
    residual_tol: float = 1e-6
    under_relaxation: float = 1.0
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

    epochs_adam: int = 5000
    epochs_lbfgs: int = 0
    max_retries: int = 1
    DNNtol: float = 1e-4
    spotter_skip: int = 1
    run_snipper: bool = True

    initial_condition: Dict = field(default_factory=dict)
    fitter_training: Dict = field(default_factory=dict)
    time_scheme: str = "implicit_euler"
    time_scheme_params: Dict = field(default_factory=dict)
    eq_L1: List[str] = field(default_factory=list)
    eq_L2: List[str] = field(default_factory=list)
    eq_F: List[str] = field(default_factory=list)
    eq_N: List[str] = field(default_factory=list)
    eq_linear_list: List[str] = field(default_factory=list)
    deriv_orders: List[int] = field(default_factory=list)
    max_deriv_orders: List[int] = field(default_factory=list)
    eq_nonlinear_list: List[str] = field(default_factory=list)
    all_derivatives: List[List[int]] = field(default_factory=list)
    f_L2: List[str] = field(default_factory=list, init=False)
    N: List[str] = field(default_factory=list, init=False)

    def _solver_specific_init(self):
        self._normalize_operator_splitting()
        self._load_pseudo_time_config()
        self._load_initial_condition_config()
        self._load_fitter_training_config()
        self.results_dir = self.get_results_dir()
        self._auto_code()
        self._parse_operator_splitting()

    def _calculate_n_eqs(self):
        return len(self.vars_list)
