"""Configuration for general PDE solver with pseudo-time stepping"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import os
import json
from src.meta_coding import parse_operators
from src.meta_coding.auto_spotter import update_physics_loss_from_config
from src.abstract_class.config.base_pde_config import BasePDEConfig


@dataclass
class PseudoTimePDEConfig(BasePDEConfig):
    """Configuration class for general PDE problems with pseudo-time stepping

    Extends BasePDEConfig with general PDE specific features:
    - Operator splitting (L1, L2, F, N) with forced existence (same as TimePDE)
    - Pseudo-time stepping parameters (instead of physical time T and dt)
    - Initial condition handling (Spotter training or given)
    - Convergence monitoring and adaptive time step control
    """

    # Pseudo-time stepping parameters (replace T and dt from TimePDE)
    max_iterations: int = 1000
    residual_tol: float = 1e-6
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

    # Training parameters (similar to TimePDE)
    epochs_adam: int = 5000
    epochs_lbfgs: int = 0
    max_retries: int = 1
    DNNtol: float = 1e-4
    spotter_skip: int = 1  # Neural network training frequency (1 = every iteration)

    # Initial condition configuration (Spotter or given)
    initial_condition: Dict = field(default_factory=dict)

    # Fitter training parameters (for neural network training during iteration)
    fitter_training: Dict = field(default_factory=dict)

    # Time integration scheme (default to implicit Euler for steady-state)
    time_scheme: str = "implicit_euler"
    time_scheme_params: Dict = field(default_factory=dict)

    # Operator splitting configuration (same as TimePDE)
    eq_L1: List[str] = field(default_factory=list)
    eq_L2: List[str] = field(default_factory=list)
    eq_F: List[str] = field(default_factory=list)
    eq_N: List[str] = field(default_factory=list)

    # Legacy backward compatibility fields
    eq_linear_list: List[str] = field(default_factory=list)
    deriv_orders: List[int] = field(default_factory=list)
    max_deriv_orders: List[int] = field(default_factory=list)
    eq_nonlinear_list: List[str] = field(default_factory=list)
    all_derivatives: List[List[int]] = field(default_factory=list)

    # Backward compatibility aliases
    f_L2: List[str] = field(default_factory=list, init=False)
    N: List[str] = field(default_factory=list, init=False)

    def _solver_specific_init(self):
        """Pseudo-Time PDE specific initialization"""
        # Extract operator splitting from eq field (same as TimePDE)
        self._normalize_operator_splitting()

        # Load pseudo-time stepping parameters
        self._load_pseudo_time_config()

        # Load initial condition config
        self._load_initial_condition_config()

        # Load fitter training config
        self._load_fitter_training_config()

        # Set results directory
        self.results_dir = self.get_results_dir()

        # Perform auto-code generation if enabled
        self._auto_code()

        # Parse equations with operator splitting (same as TimePDE)
        self._parse_operator_splitting()

    def _calculate_n_eqs(self):
        """Calculate number of equations for general PDE

        For general PDEs, n_eqs equals the number of variables since
        L1, L2, F, N are different operator terms of the same PDE.
        """
        return len(self.vars_list)

    def _normalize_operator_splitting(self):
        """Normalize equation format and ensure all operators exist (forced existence)

        Identical to TimePDEConfig implementation
        """
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

    def _load_pseudo_time_config(self):
        """Load pseudo-time stepping parameters from config

        Priority:
        1. Nested "pseudo_time" block (recommended, user's design)
        2. Nested "pseudo_time_stepping" block (legacy, backward compatible)
        3. Top-level fields (lowest priority, fallback)
        """
        # Try to load from attributes set by load_config_from_json_relaxed
        # Priority 1: Check for nested "pseudo_time" block (user's design)
        pts_params = getattr(self, 'pseudo_time', {})
        config_source = "nested 'pseudo_time'"

        # Priority 2: Fall back to "pseudo_time_stepping" block (legacy)
        if not pts_params:
            pts_params = getattr(self, 'pseudo_time_stepping', {})
            config_source = "'pseudo_time_stepping' (legacy)" if pts_params else "default values"

        # Warn if T/dt found in config (common mistake)
        if hasattr(self, 'T') or hasattr(self, 'dt'):
            print("⚠ Warning: Found 'T' or 'dt' in general_pde config (will be ignored)")
            print("  For general_pde, use 'pseudo_time' block instead:")
            print("  \"pseudo_time\": {\"max_iterations\": ..., \"dt_init\": ..., ...}")

        # Update from nested config or fall back to top-level or defaults
        self.max_iterations = pts_params.get("max_iterations",
                                            getattr(self, "max_iterations", self.max_iterations))
        self.residual_tol = pts_params.get("residual_tol",
                                          getattr(self, "residual_tol", self.residual_tol))
        self.dt_init = pts_params.get("dt_init",
                                     getattr(self, "dt_init", self.dt_init))
        self.dt_min = pts_params.get("dt_min",
                                    getattr(self, "dt_min", self.dt_min))
        self.dt_max = pts_params.get("dt_max",
                                    getattr(self, "dt_max", self.dt_max))
        self.adaptive_dt = pts_params.get("adaptive_dt",
                                         getattr(self, "adaptive_dt", self.adaptive_dt))
        self.dt_increase_factor = pts_params.get("dt_increase_factor",
                                                 getattr(self, "dt_increase_factor", self.dt_increase_factor))
        self.dt_decrease_factor = pts_params.get("dt_decrease_factor",
                                                 getattr(self, "dt_decrease_factor", self.dt_decrease_factor))
        self.check_interval = pts_params.get("check_interval",
                                            getattr(self, "check_interval", self.check_interval))
        self.print_interval = pts_params.get("print_interval",
                                            getattr(self, "print_interval", self.print_interval))
        self.save_interval = pts_params.get("save_interval",
                                           getattr(self, "save_interval", self.save_interval))
        self.max_residual = pts_params.get("max_residual",
                                          getattr(self, "max_residual", self.max_residual))
        self.residual_stagnation_steps = pts_params.get("residual_stagnation_steps",
                                                        getattr(self, "residual_stagnation_steps", self.residual_stagnation_steps))

        # Also read time_scheme from nested block if present
        if "time_scheme" in pts_params:
            self.time_scheme = pts_params["time_scheme"]
        elif hasattr(self, "time_scheme") and self.time_scheme != "implicit_euler":
            # Keep the time_scheme from top-level if it was set
            pass

        print(f"✓ Pseudo-time stepping parameters (from {config_source}):")
        print(f"  time_scheme: {self.time_scheme}")
        print(f"  max_iterations: {self.max_iterations}")
        print(f"  residual_tol: {self.residual_tol:.2e}")
        print(f"  dt_init: {self.dt_init}, dt_range: [{self.dt_min}, {self.dt_max}]")
        print(f"  adaptive_dt: {self.adaptive_dt}")

    def _load_initial_condition_config(self):
        """Load initial condition configuration"""
        # Default initial condition config
        default_ic = {
            "method": "spotter",  # or "given", "zero"
            "use_periodic": False,
            "spotter_config": {
                "epochs": 10000,
                "learning_rate": 0.01,
                "n_pde_points": 5000,
                "optimizer": "adam",
                "target_residual": 0.01,
                "print_interval": 200,
                "early_stopping": True,
                "patience": 500,
                "max_retries": 1
            }
        }

        # Update from config file (already loaded as attribute)
        if not hasattr(self, 'initial_condition') or not self.initial_condition:
            self.initial_condition = default_ic
        elif isinstance(self.initial_condition, dict):
            # Merge with defaults
            for key, value in default_ic.items():
                if key not in self.initial_condition:
                    self.initial_condition[key] = value

        # Ensure spotter_config exists
        if "spotter_config" not in self.initial_condition:
            self.initial_condition["spotter_config"] = default_ic["spotter_config"]

        print(f"Initial condition method: {self.initial_condition.get('method', 'spotter')}")
        spotter_cfg = self.initial_condition.get("spotter_config", {})
        print(f"  Spotter epochs: {spotter_cfg.get('epochs', 'N/A')}")
        print(f"  Spotter learning rate: {spotter_cfg.get('learning_rate', 'N/A')}")

    def _load_fitter_training_config(self):
        """Load fitter training configuration for neural network updates during iteration

        Note: This is only for backward compatibility. New configs should use main-level
        epochs_adam, DNNtol, etc. directly.
        """
        # Check if user provided a fitter_training block
        has_fitter_block = hasattr(self, 'fitter_training') and self.fitter_training

        if not has_fitter_block:
            # No fitter_training block - use main config values (don't override!)
            self.fitter_training = {}
        elif isinstance(self.fitter_training, dict):
            # User provided fitter_training block - merge with main config
            # Priority: fitter_training block > main config > class defaults

            # Only override main config if fitter_training block has explicit values
            if "epochs" in self.fitter_training:
                self.epochs_adam = self.fitter_training["epochs"]

            if "epochs_lbfgs" in self.fitter_training:
                self.epochs_lbfgs = self.fitter_training["epochs_lbfgs"]

            if "DNNtol" in self.fitter_training:
                self.DNNtol = self.fitter_training["DNNtol"]

        print(f"Fitter training parameters (from {'fitter_training block' if has_fitter_block else 'main config'}):")
        print(f"  epochs_adam: {self.epochs_adam}")
        print(f"  epochs_lbfgs: {self.epochs_lbfgs}")
        print(f"  DNNtol: {self.DNNtol:.2e}")

    def _auto_code(self):
        """Auto-code generation for general PDEs (same as TimePDE)"""
        if hasattr(self, "auto_code") and self.auto_code:
            try:
                # Use the new unified entrypoint that reads config.json
                config_path = os.path.join(self.case_dir, "config.json")
                update_physics_loss_from_config(config_path)

                # Disable auto_code in config
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
        """Parse equations with operator splitting (same as TimePDE)"""
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
            print("Successfully parsed equations for general PDE solver")
        except Exception as e:
            print(f"Warning: Failed to parse operator splitting: {e}")
            self.operator_parse = {}

    def _int_list_fields(self):
        """Additional integer list fields specific to general PDE"""
        base_fields = super()._int_list_fields()
        return base_fields + [
            "max_iterations", "check_interval", "print_interval", "save_interval",
            "residual_stagnation_steps", "epochs_adam", "epochs_lbfgs",
            "max_retries", "spotter_skip"
        ]

    def validate(self) -> bool:
        """Validate configuration parameters"""
        # Check pseudo-time stepping parameters
        if self.max_iterations <= 0:
            print(f"Error: max_iterations must be positive, got {self.max_iterations}")
            return False

        if self.residual_tol <= 0:
            print(f"Error: residual_tol must be positive, got {self.residual_tol}")
            return False

        if self.dt_min <= 0:
            print(f"Error: dt_min must be positive, got {self.dt_min}")
            return False

        if not (self.dt_min <= self.dt_init <= self.dt_max):
            print(f"Error: dt_init must satisfy dt_min <= dt_init <= dt_max")
            print(f"  Got: dt_min={self.dt_min}, dt_init={self.dt_init}, dt_max={self.dt_max}")
            return False

        return True

    def get_summary(self) -> Dict:
        """Get configuration summary"""
        summary = super().get_summary()
        summary.update({
            "pseudo_time_stepping": {
                "max_iterations": self.max_iterations,
                "residual_tol": self.residual_tol,
                "dt_init": self.dt_init,
                "dt_range": [self.dt_min, self.dt_max],
                "adaptive_dt": self.adaptive_dt,
                "time_scheme": self.time_scheme
            },
            "initial_condition": self.initial_condition,
            "fitter_training": self.fitter_training
        })
        return summary
