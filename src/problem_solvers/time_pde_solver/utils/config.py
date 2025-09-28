from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import numpy as np
import os
import json
from src.meta_coding import parse_operators
from src.meta_coding.auto_spotter import update_hybrid_fitter_code
from src.meta_coding.auto_spotter import update_physics_loss_from_config
from src.abstract_class.config.base_config import BaseConfig

@dataclass
class TimePDEConfig(BaseConfig):
    """Configuration class for time-dependent PDE problems"""

    # Required fields
    case_dir: str
    vars_list: List[str] = field(default_factory=list)
    spatial_vars: List[str] = field(default_factory=list)
    eq: Union[List[str], Dict[str, List[str]]] = field(default_factory=list)
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

    # Time推EnterConfiguration - Fromconfig.jsonRead
    time_scheme: str = "IMEX_RK_2_2_2"
    time_scheme_params: Dict = field(default_factory=dict)

    # OperatorsSeparateConfiguration - Fromconfig.json的eqfieldAnalyticalGet
    eq_L1: List[str] = field(default_factory=list)      # 主Linear operators
    eq_L2: List[str] = field(default_factory=list)      # 半隐式Linear operators  
    eq_F: List[str] = field(default_factory=list)       # 对应L2的Nonlinearfunction(F)
    eq_N: List[str] = field(default_factory=list)       # CompletelyNonlinearItem(N)

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

    # AnalyticalBackward的EquationList
    eq_linear_list: List[str] = field(default_factory=list)
    deriv_orders: List[int] = field(default_factory=list)
    max_deriv_orders: List[int] = field(default_factory=list)
    eq_nonlinear_list: List[str] = field(default_factory=list)
    all_derivatives: List[List[int]] = field(default_factory=list)

    # TimeformatRunning时Parameter
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

        # Normalize eq format and extract operator splitting
        self._normalize_eq_format()

        # Setup time scheme parameters
        self._setup_time_scheme_params()

        # Validate configuration
        self._validate_config()

        # Initialize其他Parameter
        self.n_dim = len(self.spatial_vars)
        self.n_eqs = self._determine_n_eqs()
        print(f"Debug: n_dim = {self.n_dim}, n_eqs = {self.n_eqs}, vars_list = {self.vars_list}")
        self._init_segment_ranges()
        self._init_boundaries()
        self.init_seed()
        self.DNN_degree = self.hidden_dims[-1]
        
        # SetupResultDirectory
        self.results_dir = os.path.join(self.case_dir, "results")
        
        self._auto_code()

        # Parse equations with operator splitting
        self._parse_operator_splitting()

    def _normalize_eq_format(self):
        """StandardizationEquationformat并提取OperatorsSeparateinformation"""
        if isinstance(self.eq, dict):
            # FromDictionaryformat提取OperatorsSeparateinformation
            self.eq_L1 = self.eq.get("L1", [])
            self.eq_L2 = self.eq.get("L2", [])
            self.eq_F = self.eq.get("F", [])  # JSON中的Ffield
            self.eq_N = self.eq.get("N", [])
            
            # MaintainTowardBackward兼容的别名
            self.f_L2 = self.eq_F  # 为TowardBackward兼容保留f_L2别名
            self.N = self.eq_N     # 为TowardBackward兼容保留N别名
            
            print(f"Extracted operator splitting from config:")
            print(f"  L1 (主Linear operators): {self.eq_L1}")
            print(f"  L2 (半隐式Linear operators): {self.eq_L2}")
            print(f"  F (Nonlinearfunction): {self.eq_F}")
            print(f"  N (CompletelyNonlinearItem): {self.eq_N}")
        elif isinstance(self.eq, list):
            # 兼容OldListformat，将其作为L1Operators
            self.eq_L1 = self.eq
            self.eq_L2 = []
            self.eq_F = []
            self.eq_N = []
            self.f_L2 = []
            self.N = []
            print(f"Using legacy list format as L1 operators: {self.eq_L1}")
        else:
            raise ValueError(f"Invalid eq format: {type(self.eq)}. Must be list or dict.")

    def _determine_n_eqs(self):
        """According toDifferent的Configuration方式确定Number of equations量 - ForTimePDE，Number of equations量等于variable数量"""
        # ForTimePDE，Number of equations量就Yesvariable数量，BecauseL1, L2, F, NYes同一个PDE的DifferentOperatorsItem
        return len(self.vars_list)

    def _setup_time_scheme_params(self):
        """SetupTimeformatParameter"""
        # SetupDefaultParameter
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

        # GetDefaultParameter
        defaults = scheme_defaults.get(self.time_scheme, {})
        
        # MergeUserParameter和DefaultParameter
        merged_params = {**defaults, **self.time_scheme_params}

        # Setup为classattribute
        for key, value in merged_params.items():
            setattr(self, key, value)

        print(f"Time scheme: {self.time_scheme}")
        print(f"Time scheme parameters: {merged_params}")

    def _parse_operator_splitting(self):
        """AnalyticalOperatorsSeparateConfiguration"""
        print("Parsing operator splitting...")
        print(f"L1 operators: {self.eq_L1}")
        print(f"L2 operators: {self.eq_L2}")
        print(f"F functions: {self.eq_F}")
        print(f"N operators: {self.eq_N}")

        # BuildDictionaryformat的Operation符用于Analytical
        if isinstance(self.eq, dict):
            # ForDictionaryformat，Re-BuildOperation符Dictionary
            operators_dict = {}
            
            # 添加L1Operators（主Linear operators）
            if self.eq_L1:
                operators_dict['L1'] = self.eq_L1
            
            # 添加L2Operators（半隐式Linear operators）
            if self.eq_L2:
                operators_dict['L2'] = self.eq_L2
            
            # 添加FOperators（IMEX-RKNeed单独的Ffunction）
            if self.eq_F:
                operators_dict['F'] = self.eq_F
            
            # Process半隐式Item L2 * F
            if self.eq_L2 and self.eq_F:
                semi_implicit_terms = []
                for i, (l2_op, f_func) in enumerate(zip(self.eq_L2, self.eq_F)):
                    # Construct半隐式Item：L2_i * F_i
                    semi_implicit_term = f"{l2_op}*({f_func})"
                    semi_implicit_terms.append(semi_implicit_term)
                    print(f"Generated semi-implicit term {i+1}: {semi_implicit_term}")
                operators_dict['L2F'] = semi_implicit_terms
            
            # 添加CompletelyNonlinearItem
            if self.eq_N:
                operators_dict['N'] = self.eq_N
            
            # UsingDictionaryformatEnter行Analytical
            self.operator_parse = parse_operators(operators_dict, self.vars_list, self.spatial_vars, self.const_list)
        else:
            # 兼容旧format - 将ListConvert为Dictionaryformat
            if isinstance(self.eq, list):
                operators_dict = {'eq': self.eq}
            else:
                operators_dict = self.eq
            self.operator_parse = parse_operators(operators_dict, self.vars_list, self.spatial_vars, self.const_list)

        # OutputAnalyticalResult并SetupConfigurationattribute
        if hasattr(self, 'operator_parse'):
            print(f"Successfully parsed equations for time PDE solver")
            
            # Ifoperator_parseYesDictionary，Setup为objectattribute
            if isinstance(self.operator_parse, dict):
                # Create一个简单objectComeStoreAnalyticalResult
                class ParseResult:
                    pass
                
                parse_obj = ParseResult()
                for key, value in self.operator_parse.items():
                    setattr(parse_obj, key, value)
                self.operator_parse = parse_obj
            
            # Setupbase_fitterNeed的attribute
            if hasattr(self.operator_parse, 'max_derivative_orders'):
                self.max_derivative_orders = self.operator_parse.max_derivative_orders
            if hasattr(self.operator_parse, 'all_derivatives'):
                self.all_derivatives = self.operator_parse.all_derivatives
            if hasattr(self.operator_parse, 'derivatives'):
                self.derivatives = self.operator_parse.derivatives
            if hasattr(self.operator_parse, 'operator_terms'):
                self.operator_terms = self.operator_parse.operator_terms
                
            print(f"Set config attributes from operator_parse:")
            print(f"  max_derivative_orders: {getattr(self, 'max_derivative_orders', 'MISSING')}")
            print(f"  all_derivatives: {getattr(self, 'all_derivatives', 'MISSING')}")
            print(f"  derivatives: {getattr(self, 'derivatives', 'MISSING')}")
            print(f"  operator_terms: {getattr(self, 'operator_terms', 'MISSING')}")

    def _auto_code(self):
        """自动代yardGenerate"""
        if hasattr(self, "auto_code") and self.auto_code:
            print("Starting auto code generation for time PDE solver...")
            
            # Based onOperatorsSeparateBuildEquationList
            linear_eqs = []
            nonlinear_eqs = []
            
            # 添加主Linear operators(L1)
            linear_eqs.extend(self.eq_L1)
            
            # 添加半隐式Linear operators(L2)
            linear_eqs.extend(self.eq_L2)
            
            # 添加CompletelyNonlinearItem(N)
            nonlinear_eqs.extend(self.eq_N)
            
            # Process半隐式Item L2 * F
            if self.eq_L2 and self.eq_F:
                for l2_op, f_func in zip(self.eq_L2, self.eq_F):
                    semi_implicit_term = f"{l2_op}*({f_func})"
                    nonlinear_eqs.append(semi_implicit_term)
                    print(f"Added semi-implicit term: {semi_implicit_term}")
            
            print(f"Auto code generation with:")
            print(f"  Linear equations: {linear_eqs}")
            print(f"  Nonlinear equations: {nonlinear_eqs}")
            
            # 调用代yardGenerate
            config_path = os.path.join(self.case_dir, "config.json")
            update_physics_loss_from_config(config_path)

            # 自动将auto_codeSetup为false
            self._disable_auto_code(config_path)
            print("Auto code generation completed, please check the net.py file, restart the program")

            # ExitProcedure
            import sys
            sys.exit(0)

    def _disable_auto_code(self, config_path: str):
        """将ConfigurationFile中的auto_codeSetup为false"""
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            config['auto_code'] = False

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

            print(f"自动将 auto_code Setup为 false AtFile: {config_path}")

        except Exception as e:
            print(f"无法UpdateConfigurationFile: {e}")

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

    def _int_list_fields(self):
        """List of fields that need to be converted to integers"""
        return ["n_segments", "poly_degree", "hidden_dims"]

    def _list_fields(self):
        """List of fields that need special handling"""
        return ["n_segments", "poly_degree", "hidden_dims", "x_domain", 
                "eq_L1", "eq_L2", "eq_F", "eq_N", "vars_list", "spatial_vars", "const_list"]

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
