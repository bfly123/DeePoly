from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import os

from src.abstract_class.config.base_config import BaseConfig
from src.abstract_class.config.base_pde_config_loading_mixin import (
    BasePDEConfigLoadingMixin,
)
from src.abstract_class.config.base_pde_config_validation_mixin import (
    BasePDEConfigValidationMixin,
)


@dataclass
class BasePDEConfig(
    BasePDEConfigValidationMixin,
    BasePDEConfigLoadingMixin,
    BaseConfig,
):
    """Shared initialization flow for PDE-style solver configs."""

    case_dir: str

    vars_list: List[str] = field(default_factory=list)
    spatial_vars: List[str] = field(default_factory=list)
    eq: Union[List[str], Dict[str, List[str]]] = field(default_factory=list)
    eq_nonlinear: List[str] = field(default_factory=list)
    const_list: List[str] = field(default_factory=list)

    n_segments: List[int] = field(default_factory=lambda: [10])
    poly_degree: List[int] = field(default_factory=lambda: [3])
    x_domain: List = field(default_factory=list)

    method: str = "hybrid"
    linear_method: str = "auto"
    linear_rcond: Optional[float] = None
    linear_alpha: Optional[float] = None
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    device: str = "cuda"
    linear_device: str = "cpu"
    learning_rate: float = 0.001
    spotter_feature_backend: str = "torch"
    spotter_feature: Dict = field(default_factory=dict)
    spotter: Dict = field(default_factory=dict)

    points_domain: int = 1000
    points_domain_test: int = 200
    points_per_swap: int = 1

    seed: int = 42
    boundary_conditions: List[dict] = field(default_factory=list)
    boundary_constraint_weight: float = 10.0
    periodic_constraint_weight: float = 10.0
    swap_constraint_weight: float = 10.0

    n_dim: int = field(init=False)
    n_eqs: int = field(init=False)
    operator_parse: Dict = field(default_factory=dict, init=False)
    DNN_degree: int = field(init=False)
    constants: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        BaseConfig.__init__(self, self.case_dir)
        self.load_config_from_json_relaxed(self.case_dir)
        self._normalize_eq_format()
        self._validate_pde_config()
        self.constants = self._merge_constants()
        self.n_dim = len(self.spatial_vars)
        self.n_eqs = self._calculate_n_eqs()
        self._init_segment_ranges()
        self._init_boundaries()
        self.init_seed()
        self.DNN_degree = self.hidden_dims[-1]
        self._parse_operators()
        self._solver_specific_init()

    def _int_list_fields(self):
        return ["n_segments", "poly_degree", "hidden_dims", "points_domain_test"]

    def _list_fields(self):
        return ["x_domain", "vars_list", "spatial_vars", "const_list", "eq_nonlinear"]

    def _process_list_field(self, key, value):
        if key == "x_domain":
            if value and isinstance(value[0], (int, float)):
                return [[value[index], value[index + 1]] for index in range(0, len(value), 2)]
        return value

    def _solver_specific_init(self):
        return None

    def _merge_constants(self) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        if isinstance(self.const_list, dict):
            return dict(self.const_list)
        if isinstance(self.const_list, list):
            for item in self.const_list:
                if isinstance(item, dict):
                    merged.update(item)
        return merged

    def get_results_dir(self):
        if hasattr(self, "results_dir") and self.results_dir:
            return self.results_dir
        return os.path.join(self.case_dir, "results")

    def _auto_code_common(self):
        if hasattr(self, "auto_code") and self.auto_code:
            print("Auto-code generation enabled")
        else:
            print("Auto-code generation disabled")
