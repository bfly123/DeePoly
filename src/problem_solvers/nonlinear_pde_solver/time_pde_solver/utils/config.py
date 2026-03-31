from dataclasses import dataclass, field
from typing import Dict, List

from src.abstract_class.config.base_pde_config import BasePDEConfig
from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.time_config_misc_mixin import (
    TimeConfigMiscMixin,
)
from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.time_config_operator_mixin import (
    TimeConfigOperatorMixin,
)
from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.time_config_scheme_mixin import (
    TimeConfigSchemeMixin,
)

@dataclass
class TimePDEConfig(
    TimeConfigOperatorMixin,
    TimeConfigSchemeMixin,
    TimeConfigMiscMixin,
    BasePDEConfig,
):
    T: float = 1.0
    dt: float = 0.01
    spotter_skip: int = 10
    epochs_adam: int = 5000
    epochs_lbfgs: int = 300
    max_retries: int = 1
    DNNtol: float = 0.0001
    Initial_conditions: List[dict] = field(default_factory=list)
    time_scheme: str = "onestep_predictor"
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
    gamma: float = field(init=False)
    max_time_steps: int = field(init=False)
    time_output_interval: int = field(init=False)
    adaptive_dt: bool = field(init=False)
    cfl_number: float = field(init=False)
    stability_check: bool = field(init=False)
    max_eigenvalue_estimate: float = field(init=False)
    f_L2: List[str] = field(default_factory=list, init=False)
    N: List[str] = field(default_factory=list, init=False)

    def _solver_specific_init(self):
        self._normalize_operator_splitting()
        self._setup_time_scheme_params()
        self.results_dir = self.get_results_dir()
        self._auto_code()
        self._parse_operator_splitting()

    def _calculate_n_eqs(self):
        return len(self.vars_list)
