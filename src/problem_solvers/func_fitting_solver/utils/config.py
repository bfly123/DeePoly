from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict
import numpy as np
from src.abstract_class.config.base_config import BaseConfig
from src.meta_coding import parse_operators
import os
import json

@dataclass
class FuncFittingConfig(BaseConfig):
    """Configuration class for function fitting problems"""
    
    # Required fields
    case_dir: str
    vars_list: List[str] = field(default_factory=list)
    spatial_vars: List[str] = field(default_factory=list)
    eq: Union[List[str], Dict[str, List[str]]] = field(default_factory=list)
    eq_nonlinear: List[str] = field(default_factory=list)
    const_list: List[str] = field(default_factory=list)

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
    max_retries: int = 1

    # Data parameters
    points_domain: int = 1000
    points_domain_test: int = 200
    points_per_swap: int = 20

    # Function fitting specific parameters
    target_function_str: Optional[str] = None
    plot_module_path: Optional[str] = None
    seed: int = 42

    # Runtime computed fields
    n_dim: int = field(init=False)
    n_eqs: int = field(init=False)
    x_min: np.ndarray = field(init=False)
    x_max: np.ndarray = field(init=False)
    segment_ranges: List[np.ndarray] = field(default_factory=list, init=False)
    x_min_norm: np.ndarray = field(init=False)
    x_max_norm: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize configuration parameters"""
        BaseConfig.__init__(self, self.case_dir)
        
        # Load configuration
        self.load_config_from_json(self.case_dir)
        
        # Normalize eq format
        self._normalize_eq_format()
        
        # Process configuration field mappings
        self._map_config_fields()
        
        # Validate required parameters
        self._validate_required_params()

        # Initialize other parameters
        self.n_dim = len(self.spatial_vars)
        self.n_eqs = self._calculate_n_eqs()
        self._init_segment_ranges()
        self._init_boundaries()
        self.init_seed()
        self.DNN_degree = self.hidden_dims[-1]

        # Parse equations
        self.operator_parse = parse_operators(self.eq, self.vars_list, self.spatial_vars, self.const_list)

    def _normalize_eq_format(self):
        """Convert eq to standardized dictionary format"""
        if isinstance(self.eq, list):
            # Convert list format to dictionary format
            self.eq = {"L1": self.eq}
        elif not isinstance(self.eq, dict):
            raise ValueError(f"Invalid eq format: {type(self.eq)}. Must be list or dict.")

    def _calculate_n_eqs(self):
        """Calculate number of equations"""
        if isinstance(self.eq, dict):
            # For dictionary format, sum all equations in all operators
            total_eqs = 0
            for op_name, eq_list in self.eq.items():
                if isinstance(eq_list, list):
                    total_eqs += len(eq_list)
                else:
                    total_eqs += 1
            return max(total_eqs, len(self.vars_list))
        else:
            # Fallback
            return len(self.vars_list)

    def _map_config_fields(self):
        """Handle configuration field mappings"""
        # Map eq to Eq for backward compatibility
        if hasattr(self, 'eq') and not hasattr(self, 'Eq'):
            self.Eq = self.eq
            
        # Map eq_nonlinear to Eq_nonlinear for backward compatibility
        if hasattr(self, 'eq_nonlinear') and not hasattr(self, 'Eq_nonlinear'):
            self.Eq_nonlinear = self.eq_nonlinear
            
        # Create empty const_list if not set
        if not hasattr(self, 'const_list'):
            self.const_list = []

    def _validate_required_params(self):
        """Validate required parameters"""
        required_params = [
            "vars_list",
            "spatial_vars",
            "n_segments",
            "poly_degree",
            "x_domain",
        ]

        for param in required_params:
            if not hasattr(self, param) or getattr(self, param) is None:
                raise ValueError(f"Required parameter '{param}' is not set")

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
        return ["n_segments", "poly_degree", "hidden_dims", "x_domain"]

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
