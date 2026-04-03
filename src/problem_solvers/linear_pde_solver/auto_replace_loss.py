import re
import json
import sys
import os
from typing import List, Dict, Tuple, Optional

from src.abstract_class.config.config_io import load_config_file, resolve_case_config_path, save_config_file

from .loss_code_generator_codegen_mixin import LossCodeGeneratorCodegenMixin
from .loss_code_generator_update_mixin import LossCodeGeneratorUpdateMixin

def update_config_auto_code(config_path: str, auto_code: bool = False) -> None:
    """Update the auto_code setting in config file
    
    Args:
        config_path: Path to the config file
        auto_code: New value for auto_code
    """
    try:
        config = load_config_file(config_path)
        config['auto_code'] = auto_code
        save_config_file(config_path, config)
    except Exception as e:
        print(f"Warning: Failed to update config file: {e}")

class LossCodeGenerator(LossCodeGeneratorUpdateMixin, LossCodeGeneratorCodegenMixin):
    """Class to generate PINNs physics loss code for PDE systems"""
    
    def __init__(self, spatial_vars: List[str], vars_list: List[str], const_list: Optional[List[Dict[str, float]]] = None, model_path: Optional[str] = None):
        """Initialize the code generator
        
        Args:
            spatial_vars: List of spatial variables (e.g. ["x", "y", "t"])
            vars_list: List of dependent variables (e.g. ["u", "v", "p"])
            const_list: List of constant dictionaries (e.g. [{"Re": 100}])
            model_path: Path to the model file to update
        """
        self.dimensions = spatial_vars
        self.vars_list = vars_list
        self.const_list = const_list or []
        self.model_path = model_path or "src/problem_solvers/linear_pde_solver/core/net.py"

def update_physics_loss_code(
    linear_equations: List[str],
    nonlinear_equations: List[str] = None,
    vars_list: List[str] = None,
    spatial_vars: List[str] = None,
    const_list: List[Dict[str, float]] = None,
    model_path: Optional[str] = None,
    case_dir: Optional[str] = None,
) -> None:
    """Convenient function to update code in physics_loss function
    
    Args:
        linear_equations: Linear parts of the equations
        nonlinear_equations: Nonlinear parts of the equations (optional)
        vars_list: List of variables
        spatial_vars: List of spatial dimensions
        const_list: List of constant dictionaries
        model_path: Path to net.py, default None uses preset path
        config_path: Path to case config file, default None uses preset path
    """
    generator = LossCodeGenerator(spatial_vars, vars_list, const_list, model_path)
    generator.update_code(linear_equations, nonlinear_equations)
    
    # Update config file and exit
    if case_dir:
      config_path = resolve_case_config_path(case_dir)
      update_config_auto_code(config_path, False)
      print("Auto code completed, please check the net.py file, restart the program")
      sys.exit(0) 
