from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict
import numpy as np
import os
import json
from src.abstract_class.config.base_pde_config import BasePDEConfig

@dataclass
class FuncFittingConfig(BasePDEConfig):
    """Configuration class for function fitting problems

    Extends BasePDEConfig with function fitting specific features:
    - Target function specification
    - Plot module path configuration
    - Higher default points per swap
    - No boundary conditions (function fitting specific)
    """

    # Function fitting specific parameters
    target_function_str: Optional[str] = None
    plot_module_path: Optional[str] = None

    # Override defaults for function fitting
    points_per_swap: int = 20  # Higher default for function fitting

    def _solver_specific_init(self):
        """Function fitting specific initialization"""
        # Process configuration field mappings
        self._map_config_fields()

        # Validate function fitting specific parameters
        self._validate_func_fitting_params()

    def _map_config_fields(self):
        """Map legacy or alternative field names to standard ones"""
        # Handle any legacy field name mappings specific to function fitting
        # This can be extended as needed for backward compatibility
        pass

    def _validate_func_fitting_params(self):
        """Validate function fitting specific parameters"""
        # Check if target function is specified when needed
        if hasattr(self, 'target_function_str') and self.target_function_str:
            print(f"Target function specified: {self.target_function_str}")

        # Validate plot module path if specified
        if hasattr(self, 'plot_module_path') and self.plot_module_path:
            if not os.path.exists(self.plot_module_path):
                print(f"Warning: Plot module path does not exist: {self.plot_module_path}")

    def _int_list_fields(self):
        """Additional integer list fields specific to function fitting"""
        base_fields = super()._int_list_fields()
        return base_fields + ["max_retries"]

    def _list_fields(self):
        """Additional list fields specific to function fitting"""
        base_fields = super()._list_fields()
        return base_fields