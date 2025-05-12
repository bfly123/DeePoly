from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from src.abstract_class.config.base_config import BaseConfig
from meta_coding.auto_eq import parse_equation_to_list
import os
import json

@dataclass
class LinearPDEConfig(BaseConfig):
    # In dataclass, attributes inherited from parent class need to be explicitly declared
    case_dir: str
    
    # Add necessary fields from the configuration file
    vars_list: List[str] = field(default_factory=list)  # List of variables
    spatial_vars: List[str] = field(default_factory=list)  # Spatial variables
    eq: List[str] = field(default_factory=list)  # Equations
    eq_nonlinear: List[str] = field(default_factory=list)  # Non-linear equations
    const_list: List[str] = field(default_factory=list)  # List of constants
    source_term: bool = field(default=False)  # Whether to include source term
    
    # Basic parameters
    # n_dim is declared with init=False, do not repeat declaration here
    n_eqs: int = 1  # Number of equations (usually 1)
    n_segments: List[int] = field(default_factory=lambda: [10])  # Number of segments

    # Polynomial parameters
    poly_degree: List[int] = field(default_factory=lambda: [3])  # Polynomial degree

    # Neural network parameters
    method: str = "hybrid"  # Method selection: 'hybrid' or 'poly'
    DNN_degree: int = 10  # Neural network feature dimension (if method='hybrid')
    device: str = "cuda"  # Computing device
    linear_device: str = "cpu"  # Linear solver device
    hidden_dims: List[int] = field(
        default_factory=lambda: [64, 64, 64]
    )  # Neural network hidden layers
    learning_rate: float = 0.001  # Learning rate
    max_retries: int = 1  # Maximum number of retries
    training_epochs: int = 10000  # Number of training epochs

    # Training/testing data parameters
    n_train: int = 1000  # Total number of training points
    n_test: int = 200  # Total number of test points
    points_domain: int = 1000  # Number of domain points
    points_domain_test: int = 200  # Number of test domain points
    points_boundary: int = 200  # Number of boundary points
    points_boundary_test: int = 50  # Number of test boundary points
    points_initial: int = 0  # Number of initial points (for time-dependent problems)

    # Derivative parameters
    deriv_orders: List[List[int]] = field(
        default_factory=lambda: [[0]]
    )  # Derivative orders to fit
    all_derivatives: List[List[int]] = field(
        default_factory=lambda: [[[0]]]
    )  # Derivative orders needed for interface continuity conditions (per equation)

    # Equation parameters
    boundary_conditions: List[dict] = field(default_factory=list)  # List of boundary conditions
    
    # PDE specific parameters

    # Case specific parameters
    x_domain: List = field(default_factory=list)  # Domain boundaries
    plot_module_path: Optional[str] = None  # Relative path to custom plotting module
    seed: int = 42  # Random seed
    
    # Define fields that need to be initialized from BaseConfig
    n_dim: int = field(init=False, default=1)  # Problem dimension
    x_min: np.ndarray = field(init=False, default=None)  # Minimum coordinates for each segment
    x_max: np.ndarray = field(init=False, default=None)  # Maximum coordinates for each segment
    segment_ranges: List[np.ndarray] = field(
        default_factory=list, init=False
    )  # Segment ranges for each dimension
    x_min_norm: np.ndarray = field(init=False, default=None)  # Normalized minimum coordinates
    x_max_norm: np.ndarray = field(init=False, default=None)  # Normalized maximum coordinates
    
    def __post_init__(self):
        """Initialize configuration parameters"""
        BaseConfig.__init__(self, self.case_dir)
        
        self.load_config_from_json(self.case_dir)
        
        # Process field mappings from the configuration file (e.g., eq -> Eq)
        self._map_config_fields()
        
        # Validate required parameters
        self._validate_required_params()

        # Initialize other parameters
        self.n_dim = len(self.spatial_vars)
        self.n_eqs = len(self.eq)
        self._init_segment_ranges()
        self._init_boundaries()
        self.init_seed()
        self.DNN_degree = self.hidden_dims[-1]

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

    def _map_config_fields(self):
        """Handle mapping between field names in the configuration file and class attributes"""
        # Check if eq field exists, and if so, map it to Eq
        if hasattr(self, 'eq') and not hasattr(self, 'Eq'):
            self.Eq = self.eq
            
        # Check if eq_nonlinear field exists, and if so, map it to Eq_nonlinear
        if hasattr(self, 'eq_nonlinear') and not hasattr(self, 'Eq_nonlinear'):
            self.Eq_nonlinear = self.eq_nonlinear
            
        # If const_list is not set, create an empty list
        if not hasattr(self, 'const_list'):
            self.const_list = []

    def _validate_required_params(self):
        """Validate that required parameters are set"""
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

        # Special validations
        if len(self.spatial_vars) != len(self.n_segments):
            raise ValueError(
                f"Length of spatial_vars ({len(self.spatial_vars)}) and n_segments ({len(self.n_segments)}) do not match"
            )

        if len(self.spatial_vars) != len(self.poly_degree):
            raise ValueError(
                f"Length of spatial_vars ({len(self.spatial_vars)}) and poly_degree ({len(self.poly_degree)}) do not match"
            )

    def _init_segment_ranges(self):
        """Initialize segment ranges"""
        self.segment_ranges = []
        
        for i in range(self.n_dim):
            x_min = self.x_domain[i][0]
            x_max = self.x_domain[i][1]
            n_seg = self.n_segments[i]
            
            # Calculate segment ranges
            seg_ranges = np.linspace(x_min, x_max, n_seg + 1)
            self.segment_ranges.append(seg_ranges)
            
    def _init_boundaries(self):
        """Initialize boundary parameters"""
        # Create boundary arrays
        self.x_min = np.array([range[0] for range in self.segment_ranges])
        self.x_max = np.array([range[-1] for range in self.segment_ranges])
        
        # Normalize boundaries
        self.x_min_norm = -np.ones(self.n_dim)
        self.x_max_norm = np.ones(self.n_dim)
        
    def init_seed(self):
        """Initialize random seed"""
        np.random.seed(self.seed)
        
    # Override base class methods, define fields that need special handling
    def _int_list_fields(self):
        """List of fields that need to be converted to integers"""
        return ["n_segments", "poly_degree", "hidden_dims"]

    def _list_fields(self):
        """List of fields that need special handling"""
        return ["n_segments", "poly_degree", "hidden_dims", "x_domain"]

    def _process_list_field(self, key, value):
        """Process list type fields"""
        if key == "n_segments" or key == "poly_degree" or key == "hidden_dims":
            # Ensure it's a list of integers
            return [int(v) if isinstance(v, str) else v for v in value]
        elif key == "x_domain":
            # Ensure it's a two-dimensional list
            if isinstance(value, list) and len(value) > 0:
                if not isinstance(value[0], list):
                    # If it's a simple list [min, max], convert to [[min, max]]
                    return [value]
            return value
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