from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict
import numpy as np
import os
import json
from src.abstract_class.config.base_pde_config import BasePDEConfig
from src.meta_coding.auto_spotter import update_physics_loss_from_config

def update_physics_loss_code(linear_equations, vars_list, spatial_vars, const_list, case_dir):
    """Compatibility wrapper function"""
    config_path = os.path.join(case_dir, "config.json")
    update_physics_loss_from_config(config_path)

@dataclass
class LinearPDEConfig(BasePDEConfig):
    """Configuration class for linear PDE problems

    Extends BasePDEConfig with linear PDE specific features:
    - Source term handling
    - Boundary point sampling
    - Training epochs configuration
    - Auto-code generation for linear PDEs
    """

    # Linear PDE specific fields
    source_term: Union[bool, str] = field(default=False)
    training_epochs: int = 10000

    # Boundary sampling (specific to linear PDE)
    points_boundary: int = 200
    points_boundary_test: int = 50

    def _solver_specific_init(self):
        """Linear PDE specific initialization"""
        # Perform auto-code generation if enabled
        self._auto_code()

    def _auto_code(self):
        """Auto-code generation for linear PDEs"""
        if hasattr(self, "auto_code") and self.auto_code:
            # Convert eq to list format for auto_code
            eq_list = []
            if isinstance(self.eq, dict):
                for op_name, eq_items in self.eq.items():
                    if isinstance(eq_items, list):
                        eq_list.extend(eq_items)
                    else:
                        eq_list.append(eq_items)
            else:
                eq_list = self.eq

            update_physics_loss_code(
                linear_equations=eq_list,
                vars_list=self.vars_list,
                spatial_vars=self.spatial_vars,
                const_list=self.const_list,
                case_dir=self.case_dir
            )

            # Auto set auto_code to false
            config_path = os.path.join(self.case_dir, "config.json")
            self._disable_auto_code(config_path)
            print("Auto code completed, please check the net.py file, restart the program")

            # Exit program
            import sys
            sys.exit(0)

    def _disable_auto_code(self, config_path: str):
        """Set auto_code to false in configuration file"""
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            config['auto_code'] = False

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

            print(f"Auto set auto_code to false in file: {config_path}")

        except Exception as e:
            print(f"Cannot update configuration file: {e}")

    def _int_list_fields(self):
        """Additional integer list fields specific to linear PDE"""
        base_fields = super()._int_list_fields()
        return base_fields + ["points_boundary", "points_boundary_test", "training_epochs"]