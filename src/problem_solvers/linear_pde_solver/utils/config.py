from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from src.abstract_class.config.base_config import BaseConfig
from meta_coding.auto_eq import parse_equation_to_list
import os
import json


@dataclass
class LinearPDEConfig(BaseConfig):
    """Configuration class for linear PDE problems"""

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

    def __post_init__(self):
        """Initialize configuration parameters"""
        BaseConfig.__init__(self, self.case_dir)

        # Load configuration
        self.load_config_from_json(self.case_dir)

        # Validate configuration
        self._validate_config()

        # Initialize parameters
        self.n_dim = len(self.spatial_vars)
        self.n_eqs = len(self.eq)
        self._init_segment_ranges()
        self._init_boundaries()
        self.init_seed()

        # Parse equations
        (
            self.eq_linear_list,
            self.deriv_orders,
            self.max_deriv_orders,
            self.eq_nonlinear_list,
            self.all_derivatives,
        ) = parse_equation_to_list(
            self.eq,
            self.eq_nonlinear,
            self.vars_list,
            self.spatial_vars,
            self.const_list,
        )

    def _auto_code(self):
        if self.auto_code:
          self.auto_code_spotter()
          self.auto_code_snipper()
        pass

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
        """Load configuration from a JSON file and update object attributes

        Different from BaseConfig, this method dynamically adds all fields from the config file,
        even if they are not predefined in the class.

        Args:
            case_dir: Directory containing the configuration file
        """
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
