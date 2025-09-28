from dataclasses import dataclass, field
from typing import List, Dict, Union, Any
import numpy as np
import os
import json
from src.abstract_class.config.base_config import BaseConfig
from src.meta_coding import parse_operators

@dataclass
class BasePDEConfig(BaseConfig):
    """Base configuration class for PDE problems

    This class consolidates common PDE configuration logic from Linear, Time, and Function Fitting solvers.
    It provides unified equation normalization, field validation, and operator parsing.
    """

    # Required case directory
    case_dir: str

    # Core PDE fields (common to all PDE types)
    vars_list: List[str] = field(default_factory=list)
    spatial_vars: List[str] = field(default_factory=list)
    eq: Union[List[str], Dict[str, List[str]]] = field(default_factory=list)
    eq_nonlinear: List[str] = field(default_factory=list)
    const_list: List[str] = field(default_factory=list)

    # Basic parameters (common to all solvers)
    n_segments: List[int] = field(default_factory=lambda: [10])
    poly_degree: List[int] = field(default_factory=lambda: [3])
    x_domain: List = field(default_factory=list)

    # Neural network parameters (common to all solvers)
    method: str = "hybrid"
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    device: str = "cuda"
    linear_device: str = "cpu"
    learning_rate: float = 0.001

    # Data parameters (common to all solvers)
    points_domain: int = 1000
    points_domain_test: int = 200
    points_per_swap: int = 1

    # Other common parameters
    seed: int = 42
    boundary_conditions: List[dict] = field(default_factory=list)

    # Runtime computed fields
    n_dim: int = field(init=False)
    n_eqs: int = field(init=False)
    operator_parse: Dict = field(default_factory=dict, init=False)
    DNN_degree: int = field(init=False)

    def __post_init__(self):
        """Initialize PDE configuration parameters with unified logic"""
        # Initialize base configuration
        BaseConfig.__init__(self, self.case_dir)

        # Load configuration with relaxed JSON loading (allows unknown fields)
        self.load_config_from_json_relaxed(self.case_dir)

        # Normalize equation format (unified across all PDE solvers)
        self._normalize_eq_format()

        # Validate common PDE fields
        self._validate_pde_config()

        # Initialize computed fields
        self.n_dim = len(self.spatial_vars)
        self.n_eqs = self._calculate_n_eqs()

        # Initialize geometric parameters
        self._init_segment_ranges()
        self._init_boundaries()

        # Initialize random seed
        self.init_seed()

        # Set DNN degree
        self.DNN_degree = self.hidden_dims[-1]

        # Parse operators (unified interface)
        self._parse_operators()

        # Call solver-specific initialization
        self._solver_specific_init()

    def _normalize_eq_format(self):
        """Convert eq to standardized dictionary format

        Unified equation format normalization for all PDE solvers:
        - list -> {"L1": list} (for linear PDE and func fitting)
        - dict -> preserved as-is (for time PDE with operator splitting)
        """
        if isinstance(self.eq, list):
            # Convert list format to dictionary format
            self.eq = {"L1": self.eq}
        elif isinstance(self.eq, dict):
            # Already in dictionary format, keep as-is
            pass
        else:
            raise ValueError(f"Invalid eq format: {type(self.eq)}. Must be list or dict.")

    def _calculate_n_eqs(self):
        """Calculate number of equations (unified logic)

        Returns:
            int: Number of equations based on variables or equation count
        """
        if isinstance(self.eq, dict):
            # For dictionary format, sum all equations in all operators
            total_eqs = 0
            for op_name, eq_list in self.eq.items():
                if isinstance(eq_list, list):
                    total_eqs += len(eq_list)
                else:
                    total_eqs += 1 if eq_list else 0
            return max(total_eqs, len(self.vars_list))
        else:
            # Fallback to variable count
            return len(self.vars_list)

    def _validate_pde_config(self):
        """Validate common PDE configuration fields"""
        # Validate required fields
        required_fields = ['vars_list', 'spatial_vars', 'x_domain', 'n_segments', 'poly_degree']
        for field_name in required_fields:
            if not hasattr(self, field_name) or not getattr(self, field_name):
                raise ValueError(f"Required field '{field_name}' is missing or empty")

        # Validate dimension consistency
        if len(self.x_domain) != len(self.spatial_vars):
            raise ValueError(f"x_domain dimension ({len(self.x_domain)}) must match spatial_vars dimension ({len(self.spatial_vars)})")

        if len(self.n_segments) != len(self.spatial_vars):
            raise ValueError(f"n_segments dimension ({len(self.n_segments)}) must match spatial_vars dimension ({len(self.spatial_vars)})")

        if len(self.poly_degree) != len(self.spatial_vars):
            raise ValueError(f"poly_degree dimension ({len(self.poly_degree)}) must match spatial_vars dimension ({len(self.spatial_vars)})")

    def _parse_operators(self):
        """Parse operators using unified interface"""
        try:
            self.operator_parse = parse_operators(
                self.eq,
                self.vars_list,
                self.spatial_vars,
                self.const_list
            )
        except Exception as e:
            print(f"Warning: Failed to parse operators: {e}")
            self.operator_parse = {}

    def load_config_from_json_relaxed(self, case_dir=None):
        """Load configuration from JSON file with relaxed mode

        This allows unknown fields to be loaded (unlike BaseConfig's strict mode).
        Maintains compatibility with existing PDE solver behavior.

        Args:
            case_dir: Case directory path, defaults to current directory
        """
        config_path = os.path.join(case_dir, "config.json")
        if not os.path.exists(config_path):
            return False

        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            print(f"Successfully loaded configuration from {config_path}")

            # Update attributes (relaxed mode - allows unknown fields)
            for key, value in config_dict.items():
                # Skip metadata fields
                if key in ['export_time', 'device_info']:
                    continue

                # Process list fields with type conversion
                if isinstance(value, list):
                    # Handle integer lists
                    if key in self._int_list_fields():
                        try:
                            value = [int(v) if isinstance(v, str) else v for v in value]
                        except ValueError:
                            pass
                    # Handle special lists
                    elif key in self._list_fields():
                        value = self._process_list_field(key, value)

                # Convert string numbers to appropriate types
                elif isinstance(value, str):
                    # Try to convert string numbers
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string

                # Set attribute (relaxed - create new attributes if needed)
                setattr(self, key, value)

            return True

        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False

    def _int_list_fields(self):
        """Fields that should be converted to integer lists"""
        return ['n_segments', 'poly_degree', 'hidden_dims', 'points_domain_test']

    def _list_fields(self):
        """Fields that need special list processing"""
        return ['x_domain', 'vars_list', 'spatial_vars', 'const_list', 'eq_nonlinear']

    def _process_list_field(self, key, value):
        """Process special list fields"""
        if key == 'x_domain':
            # Ensure x_domain is a list of lists
            if value and isinstance(value[0], (int, float)):
                # Convert flat list to list of ranges
                return [[value[i], value[i+1]] for i in range(0, len(value), 2)]

        return value

    def _solver_specific_init(self):
        """Hook for solver-specific initialization

        To be overridden by subclasses for their specific initialization logic.
        """
        pass

    def get_results_dir(self):
        """Get results directory path (unified utility)"""
        if hasattr(self, 'results_dir') and self.results_dir:
            return self.results_dir
        else:
            return os.path.join(self.case_dir, "results")

    def _auto_code_common(self):
        """Common auto-code functionality

        This provides base auto-code support that can be extended by subclasses.
        """
        if hasattr(self, 'auto_code') and self.auto_code:
            print("Auto-code generation enabled")
            # Base auto-code logic can be added here
        else:
            print("Auto-code generation disabled")