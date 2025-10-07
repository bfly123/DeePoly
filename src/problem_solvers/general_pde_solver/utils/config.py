"""Configuration for general PDE solver with pseudo-time stepping"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from src.abstract_class.config.base_pde_config import BasePDEConfig


@dataclass
class GeneralPDEConfig(BasePDEConfig):
    """Configuration for general PDE solver with pseudo-time stepping

    Extends BasePDEConfig with pseudo-time stepping specific parameters:
    - Convergence tolerance
    - Maximum iterations
    - Adaptive time step control
    - Residual monitoring
    """

    # Initial conditions (not used for steady-state, but needed for compatibility)
    Initial_conditions: List[dict] = field(default_factory=list)

    # Results directory
    results_dir: str = "results"

    # Pseudo-time stepping parameters
    max_iterations: int = 10000
    convergence_tol: float = 1e-6
    residual_tol: float = 1e-8
    dt_init: float = 0.01
    dt_min: float = 1e-8
    dt_max: float = 1.0
    adaptive_dt: bool = True
    dt_increase_factor: float = 1.2
    dt_decrease_factor: float = 0.5
    check_interval: int = 10
    print_interval: int = 100
    max_residual: float = 1e10
    residual_stagnation_steps: int = 50

    # Spotter training parameters
    spotter_epochs: int = 1000

    def __post_init__(self):
        """Initialize configuration after dataclass initialization"""
        super().__post_init__()

        # Load config_dict from JSON file for accessing nested parameters
        import json
        import os
        config_path = os.path.join(self.case_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config_dict = config_dict  # Save for later access
        else:
            config_dict = {}
            self.config_dict = {}

        pts_params = config_dict.get("pseudo_time_stepping", {})

        # Pseudo-time stepping parameters
        self.max_iterations = pts_params.get("max_iterations", config_dict.get("max_iterations", 10000))
        self.convergence_tol = pts_params.get("convergence_tol", config_dict.get("convergence_tol", 1e-6))
        self.residual_tol = pts_params.get("residual_tol", config_dict.get("residual_tol", 1e-8))

        # Initial pseudo-time step
        self.dt_init = pts_params.get("dt_init", config_dict.get("dt_init", 0.01))
        self.dt_min = pts_params.get("dt_min", config_dict.get("dt_min", 1e-8))
        self.dt_max = pts_params.get("dt_max", config_dict.get("dt_max", 1.0))

        # Adaptive time step control
        self.adaptive_dt = pts_params.get("adaptive_dt", config_dict.get("adaptive_dt", True))
        self.dt_increase_factor = pts_params.get("dt_increase_factor", config_dict.get("dt_increase_factor", 1.2))
        self.dt_decrease_factor = pts_params.get("dt_decrease_factor", config_dict.get("dt_decrease_factor", 0.5))
        self.dt_safety_factor = pts_params.get("dt_safety_factor", config_dict.get("dt_safety_factor", 0.8))

        # Convergence monitoring
        self.check_interval = pts_params.get("check_interval", config_dict.get("check_interval", 10))
        self.print_interval = pts_params.get("print_interval", config_dict.get("print_interval", 100))
        self.save_interval = pts_params.get("save_interval", config_dict.get("save_interval", 500))

        # Divergence detection
        self.max_residual = pts_params.get("max_residual", config_dict.get("max_residual", 1e10))
        self.residual_stagnation_steps = pts_params.get("residual_stagnation_steps", config_dict.get("residual_stagnation_steps", 50))

        # Time scheme (default to implicit Euler)
        self.time_scheme = config_dict.get("time_scheme", "implicit_euler")

        # Spotter training parameters (check both locations)
        # 1. From initial_condition.spotter_config.epochs (new format)
        # 2. From top-level spotter_epochs (old format)
        initial_cond = config_dict.get("initial_condition", {})
        spotter_config = initial_cond.get("spotter_config", {})
        self.spotter_epochs = spotter_config.get("epochs",
                                                 config_dict.get("spotter_epochs", 1000))

        # Animation parameters
        self.realtime_visualization = config_dict.get("realtime_visualization", False)
        self.animation_skip = config_dict.get("animation_skip", 10)
        self.save_animation = config_dict.get("save_animation", True)

    def validate(self) -> bool:
        """Validate configuration parameters"""
        if not super().validate():
            return False

        # Check pseudo-time stepping parameters
        if self.max_iterations <= 0:
            print(f"Error: max_iterations must be positive, got {self.max_iterations}")
            return False

        if self.convergence_tol <= 0:
            print(f"Error: convergence_tol must be positive, got {self.convergence_tol}")
            return False

        if self.dt_init <= 0 or self.dt_init > self.dt_max:
            print(f"Error: dt_init must be in (0, dt_max], got {self.dt_init}")
            return False

        if self.dt_min >= self.dt_max:
            print(f"Error: dt_min must be less than dt_max")
            return False

        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        summary = super().get_summary()
        summary.update({
            "pseudo_time_stepping": {
                "max_iterations": self.max_iterations,
                "convergence_tol": self.convergence_tol,
                "residual_tol": self.residual_tol,
                "dt_init": self.dt_init,
                "dt_range": [self.dt_min, self.dt_max],
                "adaptive_dt": self.adaptive_dt,
                "time_scheme": self.time_scheme
            }
        })
        return summary