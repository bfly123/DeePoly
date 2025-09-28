from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import numpy as np
import os
import json
from src.meta_coding import parse_operators
from src.meta_coding.auto_spotter import update_hybrid_fitter_code
from src.meta_coding.auto_spotter import update_physics_loss_from_config
from src.abstract_class.config.base_pde_config import BasePDEConfig

@dataclass
class TimePDEConfig(BasePDEConfig):
    """Configuration class for time-dependent PDE problems

    Extends BasePDEConfig with time PDE specific features:
    - Operator splitting (L1, L2, F, N) with forced existence
    - Time integration scheme parameters
    - Two-stage training (Adam + LBFGS)
    - Spotter skip optimization
    - Initial conditions handling
    """

    # Time-specific parameters
    T: float = 1.0
    dt: float = 0.01
    spotter_skip: int = 10
    epochs_adam: int = 5000
    epochs_lbfgs: int = 300
    max_retries: int = 1
    DNNtol: float = 0.0001
    Initial_conditions: List[dict] = field(default_factory=list)

    # Time integration scheme configuration
    time_scheme: str = "IMEX_RK_2_2_2"
    time_scheme_params: Dict = field(default_factory=dict)

    # Operator splitting configuration (extracted from eq field)
    eq_L1: List[str] = field(default_factory=list)      # Primary linear operators
    eq_L2: List[str] = field(default_factory=list)      # Semi-implicit linear operators
    eq_F: List[str] = field(default_factory=list)       # Nonlinear functions (F)
    eq_N: List[str] = field(default_factory=list)       # Fully nonlinear terms (N)

    # Legacy backward compatibility fields
    eq_linear_list: List[str] = field(default_factory=list)
    deriv_orders: List[int] = field(default_factory=list)
    max_deriv_orders: List[int] = field(default_factory=list)
    eq_nonlinear_list: List[str] = field(default_factory=list)
    all_derivatives: List[List[int]] = field(default_factory=list)

    # Time scheme runtime parameters
    gamma: float = field(init=False)
    max_time_steps: int = field(init=False)
    time_output_interval: int = field(init=False)
    adaptive_dt: bool = field(init=False)
    cfl_number: float = field(init=False)
    stability_check: bool = field(init=False)
    max_eigenvalue_estimate: float = field(init=False)

    # Backward compatibility aliases
    f_L2: List[str] = field(default_factory=list, init=False)
    N: List[str] = field(default_factory=list, init=False)

    def _solver_specific_init(self):
        """Time PDE specific initialization"""
        # Extract operator splitting from eq field
        self._normalize_operator_splitting()

        # Setup time scheme parameters
        self._setup_time_scheme_params()

        # Set results directory
        self.results_dir = self.get_results_dir()

        # Perform auto-code generation if enabled
        self._auto_code()

        # Parse equations with operator splitting
        self._parse_operator_splitting()

    def _calculate_n_eqs(self):
        """Calculate number of equations for time PDE

        For time PDEs, n_eqs equals the number of variables since
        L1, L2, F, N are different operator terms of the same PDE.
        """
        return len(self.vars_list)

    def _normalize_operator_splitting(self):
        """Normalize equation format and ensure all operators exist (forced existence)"""
        if isinstance(self.eq, dict):
            # Extract operators from dictionary format, supplement missing ones with zero operators
            self.eq_L1 = self.eq.get("L1", ["0"])  # Default zero operator
            self.eq_L2 = self.eq.get("L2", ["0"])  # Default zero operator
            self.eq_F = self.eq.get("F", ["1"])    # Default unit operator
            self.eq_N = self.eq.get("N", ["0"])    # Default zero operator

            print(f"Extracted operator splitting from config (with forced existence):")
            print(f"  L1 (Primary linear operators): {self.eq_L1}")
            print(f"  L2 (Semi-implicit linear operators): {self.eq_L2}")
            print(f"  F (Nonlinear functions): {self.eq_F}")
            print(f"  N (Fully nonlinear terms): {self.eq_N}")

        elif isinstance(self.eq, list):
            # Legacy list format compatibility, treat as L1 operators, supplement others with zeros
            self.eq_L1 = self.eq if self.eq else ["0"]
            self.eq_L2 = ["0"]  # Default zero operator
            self.eq_F = ["1"]   # Default unit operator
            self.eq_N = ["0"]   # Default zero operator
            print(f"Using legacy list format as L1 operators: {self.eq_L1}")

            # For legacy format, maintain backward compatible aliases
            self.f_L2 = self.eq_F
            self.N = self.eq_N
            print(f"Legacy format preserved L1 operators: {self.eq_L1}")
            return  # Early return, skip variable count adjustment

        else:
            raise ValueError(f"Invalid eq format: {type(self.eq)}. Must be list or dict.")

        # Ensure all operator lists have consistent length with variable count
        n_vars = len(self.vars_list)
        if n_vars > 0:  # Only adjust when variable list is known
            self.eq_L1 = self._pad_operator_list(self.eq_L1, n_vars, "0")
            self.eq_L2 = self._pad_operator_list(self.eq_L2, n_vars, "0")
            self.eq_F = self._pad_operator_list(self.eq_F, n_vars, "1")
            self.eq_N = self._pad_operator_list(self.eq_N, n_vars, "0")

        # Maintain backward compatible aliases
        self.f_L2 = self.eq_F  # Keep f_L2 alias for backward compatibility
        self.N = self.eq_N     # Keep N alias for backward compatibility

        print(f"Normalized operators to {n_vars} variables:")
        print(f"  L1: {self.eq_L1}")
        print(f"  L2: {self.eq_L2}")
        print(f"  F: {self.eq_F}")
        print(f"  N: {self.eq_N}")

    def _pad_operator_list(self, op_list: List[str], target_length: int, default_op: str) -> List[str]:
        """Pad operator list to target length, ensuring dimensional consistency"""
        if len(op_list) == target_length:
            return op_list
        elif len(op_list) == 1:
            # Single operator extends to all variables
            return op_list * target_length
        elif len(op_list) == 0:
            # Empty list filled with default operator
            return [default_op] * target_length
        else:
            # Insufficient ones padded with default operator, excess ones truncated
            if len(op_list) < target_length:
                return op_list + [default_op] * (target_length - len(op_list))
            else:
                return op_list[:target_length]

    def _setup_time_scheme_params(self):
        """Setup time scheme parameters"""
        print(f"Time scheme: {self.time_scheme}")

        # Setup default parameters based on time scheme
        if self.time_scheme in ["IMEX_RK_2_2_2", "IMEX_222"]:
            self.gamma = 1 - 1/np.sqrt(2)  # IMEX-RK(2,2,2) parameter
            self.max_time_steps = 1000
            self.time_output_interval = 10
            self.adaptive_dt = False
            self.cfl_number = 0.5
            self.stability_check = True
            self.max_eigenvalue_estimate = 100.0
        elif self.time_scheme == "onestep_predictor":
            # Parameters for one-step predictor-corrector scheme
            self.max_time_steps = int(self.T / self.dt) + 1
            self.time_output_interval = 1
            self.adaptive_dt = False
            self.stability_check = False
        else:
            # Default parameters
            self.max_time_steps = 1000
            self.time_output_interval = 10
            self.adaptive_dt = False
            self.cfl_number = 0.5
            self.stability_check = True
            self.max_eigenvalue_estimate = 100.0

        # Override with user-specified parameters
        for key, value in self.time_scheme_params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        print(f"Time scheme parameters: {{'gamma': {getattr(self, 'gamma', 'N/A')}, 'max_time_steps': {self.max_time_steps}, 'time_output_interval': {self.time_output_interval}, 'adaptive_dt': {self.adaptive_dt}, 'cfl_number': {getattr(self, 'cfl_number', 'N/A')}, 'stability_check': {self.stability_check}, 'max_eigenvalue_estimate': {getattr(self, 'max_eigenvalue_estimate', 'N/A')}}}")

    def _auto_code(self):
        """Auto-code generation for time PDEs"""
        if hasattr(self, "auto_code") and self.auto_code:
            try:
                update_hybrid_fitter_code(self.case_dir)

                # Disable auto_code in config
                config_path = os.path.join(self.case_dir, "config.json")
                self._disable_auto_code(config_path)

                print("Auto code completed, please check the generated files and restart the program")

                # Exit program
                import sys
                sys.exit(0)

            except Exception as e:
                print(f"Auto-code generation failed: {e}")

    def _disable_auto_code(self, config_path: str):
        """Set auto_code to false in configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            config['auto_code'] = False

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

            print(f"Auto set auto_code to false in file: {config_path}")

        except Exception as e:
            print(f"Cannot update configuration file: {e}")

    def _parse_operator_splitting(self):
        """Parse equations with operator splitting"""
        print("Parsing operator splitting with forced existence...")
        print(f"L1 operators: {self.eq_L1}")
        print(f"L2 operators: {self.eq_L2}")
        print(f"F functions: {self.eq_F}")
        print(f"N operators: {self.eq_N}")

        # Generate semi-implicit terms L2*F
        l2f_terms = []
        for i, (l2, f) in enumerate(zip(self.eq_L2, self.eq_F)):
            if l2 != "0" and f != "1":
                l2f_terms.append(f"{l2}*({f})")
            elif l2 != "0" and f == "1":
                l2f_terms.append(l2)
            elif l2 == "0":
                l2f_terms.append("0")
            else:
                l2f_terms.append(f"0*({f})")
            print(f"Generated semi-implicit term {i+1}: {l2f_terms[-1]}")

        # Build unified equation dictionary for operator parsing
        unified_eq = {
            "L1": self.eq_L1,
            "L2": self.eq_L2,
            "F": self.eq_F,
            "N": self.eq_N,
            "L2F": l2f_terms  # Semi-implicit combination
        }

        try:
            self.operator_parse = parse_operators(
                unified_eq,
                self.vars_list,
                self.spatial_vars,
                self.const_list
            )
            print("Successfully parsed equations for time PDE solver")
        except Exception as e:
            print(f"Warning: Failed to parse operator splitting: {e}")
            self.operator_parse = {}

    def _int_list_fields(self):
        """Additional integer list fields specific to time PDE"""
        base_fields = super()._int_list_fields()
        return base_fields + ["epochs_adam", "epochs_lbfgs", "max_retries", "spotter_skip", "max_time_steps", "time_output_interval"]