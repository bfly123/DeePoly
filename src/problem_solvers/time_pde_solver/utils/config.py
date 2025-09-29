from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import os
import json
from src.meta_coding.auto_eq import parse_equation_to_list
from src.meta_coding.auto_repalce_nonlinear import update_hybrid_fitter_code
from src.problem_solvers.linear_pde_solver.auto_replace_loss import update_physics_loss_code
from src.abstract_class.config.base_config import BaseConfig

@dataclass
class TimePDEConfig(BaseConfig):
    """Configuration class for time-dependent PDE problems"""

    # Required fields
    case_dir: str
    vars_list: List[str] = field(default_factory=list)
    spatial_vars: List[str] = field(default_factory=list)
    eq: List[str] = field(default_factory=list)
    eq_nonlinear: List[str] = field(default_factory=list)
    const_list: List[str] = field(default_factory=list)
    source_term: bool = field(default=False)

    # Basic parameters
    n_segments: List[int] = field(default_factory=lambda: [10])
    poly_degree: List[int] = field(default_factory=lambda: [3])
    x_domain: List = field(default_factory=list)

    # Neural network parameters
    method: str = "hybrid"
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    device: str = "cuda"
    linear_device: str = "cpu"
    learning_rate: float = 0.001
    training_epochs: int = 10000

    # Data parameters
    points_domain: int = 1000
    points_domain_test: int = 200
    points_boundary: int = 200
    points_boundary_test: int = 50

    # Time-specific parameters
    T: float = 1.0
    dt: float = 0.01
    spotter_skip: int = 10
    epochs_adam: int = 5000
    epochs_lbfgs: int = 300
    max_retries: int = 1
    DNNtol: float = 0.0001
    points_per_swap: int = 1
    Initial_conditions: List[dict] = field(default_factory=list)

    # 时间推进配置 - 从config.json读取
    time_scheme: str = "IMEX_RK_2_2_2"
    time_scheme_params: Dict = field(default_factory=dict)

    # 算子分离配置 - 与config.json对应
    eq_L1: List[str] = field(default_factory=list)      # 主线性算子
    eq_L2: List[str] = field(default_factory=list)      # 半隐式线性算子  
    f_L2: List[str] = field(default_factory=list)       # 对应L2的非线性函数
    N: List[str] = field(default_factory=list)          # 完全非线性项

    # Other parameters
    seed: int = 42
    boundary_conditions: List[dict] = field(default_factory=list)

    # Runtime computed fields
    n_dim: int = field(init=False)
    n_eqs: int = field(init=False)
    x_min: np.ndarray = field(init=False)
    x_max: np.ndarray = field(init=False)
    segment_ranges: List[np.ndarray] = field(default_factory=list, init=False)
    x_min_norm: np.ndarray = field(init=False)
    x_max_norm: np.ndarray = field(init=False)

    # 解析后的方程列表
    eq_linear_list: List[str] = field(default_factory=list)
    deriv_orders: List[int] = field(default_factory=list)
    max_deriv_orders: List[int] = field(default_factory=list)
    eq_nonlinear_list: List[str] = field(default_factory=list)
    all_derivatives: List[List[int]] = field(default_factory=list)

    # 时间格式运行时参数
    gamma: float = field(init=False)
    max_time_steps: int = field(init=False)
    time_output_interval: int = field(init=False)
    adaptive_dt: bool = field(init=False)
    cfl_number: float = field(init=False)
    stability_check: bool = field(init=False)
    max_eigenvalue_estimate: float = field(init=False)

    def __post_init__(self):
        """Initialize configuration parameters"""
        BaseConfig.__init__(self, self.case_dir)

        # Load configuration
        self.load_config_from_json(self.case_dir)

        # Setup time scheme parameters
        self._setup_time_scheme_params()

        # Validate configuration
        self._validate_config()

        # 初始化其他参数
        self.n_dim = len(self.spatial_vars)
        self.n_eqs = self._determine_n_eqs()
        self._init_segment_ranges()
        self._init_boundaries()
        self.init_seed()
        self.DNN_degree = self.hidden_dims[-1]
        self._auto_code()

        # Parse equations with operator splitting
        self._parse_operator_splitting()

    def _determine_n_eqs(self):
        """根据不同的配置方式确定方程数量"""
        if self.eq_L1:
            return len(self.eq_L1)
        elif self.eq:
            return len(self.eq)
        else:
            return len(self.vars_list)

    def _setup_time_scheme_params(self):
        """设置时间格式参数"""
        # 设置默认参数
        scheme_defaults = {
            "IMEX_RK_2_2_2": {
                "gamma": (2 - np.sqrt(2)) / 2,
                "max_time_steps": 1000,
                "time_output_interval": 10,
                "adaptive_dt": False,
                "cfl_number": 0.5,
                "stability_check": True,
                "max_eigenvalue_estimate": 100.0
            },
            "BDF1": {
                "max_time_steps": 1000,
                "time_output_interval": 10,
                "adaptive_dt": False,
                "cfl_number": 0.8,
                "stability_check": True,
                "max_eigenvalue_estimate": 100.0
            },
            "BDF2": {
                "max_time_steps": 1000,
                "time_output_interval": 10,
                "adaptive_dt": False,
                "cfl_number": 0.6,
                "stability_check": True,
                "max_eigenvalue_estimate": 100.0
            }
        }

        # 获取默认参数
        defaults = scheme_defaults.get(self.time_scheme, {})
        
        # 合并用户参数和默认参数
        merged_params = {**defaults, **self.time_scheme_params}

        # 设置为类属性
        for key, value in merged_params.items():
            setattr(self, key, value)

        print(f"Time scheme: {self.time_scheme}")
        print(f"Time scheme parameters: {merged_params}")

    def _parse_operator_splitting(self):
        """解析算子分离配置"""
        print("Parsing operator splitting...")
        print(f"L1 operators: {self.eq_L1}")
        print(f"L2 operators: {self.eq_L2}")
        print(f"f_L2 functions: {self.f_L2}")
        print(f"N operators: {self.N}")

        # 使用算子分离的配置来解析方程
        all_linear_eqs = self.eq_L1 + self.eq_L2
        all_nonlinear_eqs = self.N.copy()

        # 处理半隐式项 L2 * f_L2
        if self.eq_L2 and self.f_L2:
            for l2_op, f_l2_func in zip(self.eq_L2, self.f_L2):
                # 构造半隐式项
                semi_implicit_term = f"{l2_op}*({f_l2_func})"
                all_nonlinear_eqs.append(semi_implicit_term)

        # 解析方程
        (
            self.eq_linear_list,
            self.deriv_orders,
            self.max_deriv_orders,
            self.eq_nonlinear_list,
            self.all_derivatives,
        ) = parse_equation_to_list(
            all_linear_eqs,
            all_nonlinear_eqs,
            self.vars_list,
            self.spatial_vars,
            self.const_list,
        )

        print(f"Parsed linear equations: {self.eq_linear_list}")
        print(f"Parsed nonlinear equations: {self.eq_nonlinear_list}")

    def _auto_code(self):
        """自动代码生成"""
        if hasattr(self, "auto_code") and self.auto_code:
            # 使用算子分离的配置生成代码
            linear_eqs = self.eq_L1 + self.eq_L2
            nonlinear_eqs = self.N.copy()

            # 处理半隐式项
            if self.eq_L2 and self.f_L2:
                for l2_op, f_l2_func in zip(self.eq_L2, self.f_L2):
                    nonlinear_eqs.append(f"{l2_op}*({f_l2_func})")

            update_physics_loss_code(
                linear_equations=linear_eqs,
                nonlinear_equations=nonlinear_eqs,
                vars_list=self.vars_list,
                spatial_vars=self.spatial_vars,
                const_list=self.const_list,
                case_dir=self.case_dir
            )

    def _validate_config(self):
        """Validate configuration parameters"""
        # Check required parameters
        required = [
            "vars_list",
            "spatial_vars", 
            "n_segments",
            "poly_degree",
            "x_domain",
        ]
        for param in required:
            if not hasattr(self, param) or getattr(self, param) is None:
                raise ValueError(f"Required parameter '{param}' is not set")

        # Validate time scheme
        supported_schemes = ["IMEX_RK_2_2_2", "BDF1", "BDF2", "Trapezoidal"]
        if self.time_scheme not in supported_schemes:
            raise ValueError(f"Unsupported time scheme: {self.time_scheme}. "
                           f"Supported schemes: {supported_schemes}")

        # Validate operator splitting consistency
        if self.eq_L2 and self.f_L2:
            if len(self.eq_L2) != len(self.f_L2):
                raise ValueError("Length of eq_L2 and f_L2 must match")

        # Validate array lengths
        if len(self.spatial_vars) != len(self.n_segments):
            raise ValueError("Length of spatial_vars and n_segments must match")
        if len(self.spatial_vars) != len(self.poly_degree):
            raise ValueError("Length of spatial_vars and poly_degree must match")

    def _int_list_fields(self):
        """List of fields that need to be converted to integers"""
        return ["n_segments", "poly_degree", "hidden_dims"]

    def _list_fields(self):
        """List of fields that need special handling"""
        return ["n_segments", "poly_degree", "hidden_dims", "x_domain", 
                "eq_L1", "eq_L2", "f_L2", "N"]

    def _process_list_field(self, key, value):
        """Process list type fields"""
        if key in ["n_segments", "poly_degree", "hidden_dims"]:
            return [int(v) if isinstance(v, str) else v for v in value]
        elif key == "x_domain" and isinstance(value, list) and len(value) > 0:
            if not isinstance(value[0], list):
                return [value]
        return value

    def load_config_from_json(self, case_dir=None):
        """Load configuration from a JSON file and update object attributes"""
        config_path = os.path.join(case_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_dict = json.load(f)

                # Apply values from the config dictionary to this object
                for key, value in config_dict.items():
                    # Special type handling
                    if isinstance(value, str) and key in self._int_list_fields():
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    elif isinstance(value, list) and key in self._list_fields():
                        value = self._process_list_field(key, value)

                    # Set attribute whether predefined or not
                    setattr(self, key, value)

                print(f"Successfully loaded configuration from {config_path}")
                return True
            except Exception as e:
                print(f"Error loading configuration file: {e}")
                return False
        else:
            print(f"Invalid configuration file path: {config_path}")
            return False

    def get_butcher_tableau(self):
        """获取当前时间格式的Butcher表"""
        if self.time_scheme == "IMEX_RK_2_2_2":
            gamma = self.gamma
            
            # 显式表
            explicit_tableau = {
                "c": [0, 1-gamma],
                "A": [[0, 0], [1-gamma, 0]], 
                "b": [0.5, 0.5]
            }
            
            # 隐式表
            implicit_tableau = {
                "c": [gamma, 1],
                "A": [[gamma, 0], [1-2*gamma, gamma]],
                "b": [0.5, 0.5]
            }
            
            return explicit_tableau, implicit_tableau
        else:
            raise NotImplementedError(f"Butcher tableau for {self.time_scheme} not implemented")

    def print_configuration_summary(self):
        """打印配置摘要"""
        print("\n" + "="*50)
        print("TIME PDE CONFIGURATION SUMMARY")
        print("="*50)
        print(f"Problem type: {self.problem_type}")
        print(f"Time scheme: {self.time_scheme}")
        print(f"Total time: {self.T}")
        print(f"Time step: {self.dt}")
        print(f"Spatial dimensions: {self.n_dim}")
        print(f"Number of equations: {self.n_eqs}")
        print(f"Variables: {self.vars_list}")
        print(f"Domain: {self.x_domain}")
        print("\nOperator Splitting:")
        print(f"  L1 (implicit): {self.eq_L1}")
        print(f"  L2 (semi-implicit): {self.eq_L2}")
        print(f"  f_L2 (functions): {self.f_L2}")
        print(f"  N (explicit): {self.N}")
        print("="*50 + "\n")
