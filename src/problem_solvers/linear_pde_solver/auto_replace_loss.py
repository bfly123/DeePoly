import re
import json
import sys
import os
from typing import List, Dict, Tuple, Optional
from src.abstract_class.base_net import BaseNet

def update_config_auto_code(config_path: str, auto_code: bool = False) -> None:
    """Update the auto_code setting in config file
    
    Args:
        config_path: Path to the config file
        auto_code: New value for auto_code
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config['auto_code'] = auto_code
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        print(f"Warning: Failed to update config file: {e}")

class LossCodeGenerator:
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
        
    def _parse_derivatives(self, equation: str) -> Dict[str, List[str]]:
        """Parse derivatives from equation string
        
        Args:
            equation: Equation string (e.g. "diff(u,x,2) + diff(u,y,2)")
            
        Returns:
            Dictionary mapping variables to their derivative orders
        """
        derivatives = {}
        # Find all diff() terms
        diff_pattern = r"diff\((\w+),(\w+)(?:,(\d+))?\)"
        matches = re.finditer(diff_pattern, equation)
        
        for match in matches:
            var, dim, order = match.groups()
            order = int(order) if order else 1
            
            if var not in derivatives:
                derivatives[var] = []
                
            # Always add all lower-order derivatives up to the required order
            for o in range(1, order + 1):
                d = dim * o
                if d not in derivatives[var]:
                    derivatives[var].append(d)
            
        return derivatives
        
    def _generate_derivatives_code(self, equations: List[str]) -> str:
        """Generate code for calculating derivatives
        
        Args:
            equations: List of equation strings
            
        Returns:
            Generated code for derivative calculations
        """
        # Collect all derivatives needed
        used_derivatives = {}
        for eq in equations:
            derivatives = self._parse_derivatives(eq)
            for var, dims in derivatives.items():
                if var not in used_derivatives:
                    used_derivatives[var] = []
                used_derivatives[var].extend(dims)
                
        # Remove duplicates while preserving order
        for var in used_derivatives:
            used_derivatives[var] = list(dict.fromkeys(used_derivatives[var]))
            
        code_lines = []
        
        # Add constants if any
        if self.const_list:
            code_lines.append("# Extract constants")
            for const_dict in self.const_list:
                for name, value in const_dict.items():
                    code_lines.append(f"{name} = {value}")
                    
        if code_lines:  # Add empty line if we added constants
            code_lines.append("")
            
        code_lines.append("# Extract physical quantities from output")
        for i, var in enumerate(self.vars_list):
            code_lines.append(f"{var} = output[..., {i}]")
            
        code_lines.append("")
        code_lines.append("# Calculate derivatives in each direction")
        
        # Generate first-order derivatives
        for var, dims in used_derivatives.items():
            for dim in dims:
                if len(dim) == 1:  # Only process first-order derivatives
                    dim_idx = self.dimensions.index(dim)
                    code_lines.append(f"d{var}_{dim} = self.gradients({var}, x_train)[0][..., {dim_idx}]")
                    
        # Generate second-order derivatives
        second_order_needed = False
        for var, dims in used_derivatives.items():
            for dim in dims:
                if len(dim) > 1:  # Check for second-order derivatives
                    second_order_needed = True
                    break
            if second_order_needed:
                break
                
        if second_order_needed:
            code_lines.append("")
            code_lines.append("# Calculate second-order derivatives")
            
            for var, dims in used_derivatives.items():
                for dim in dims:
                    if len(dim) == 2:  # Second-order derivatives
                        dim1_idx = self.dimensions.index(dim[0])
                        dim2_idx = self.dimensions.index(dim[1])
                        # First calculate first-order derivative
                        first_deriv = f"d{var}_{dim[0]}"
                        # Then calculate second-order derivative
                        code_lines.append(f"d{var}_{dim} = self.gradients({first_deriv}, x_train)[0][..., {dim2_idx}]")
                        
        return "\n".join(code_lines)
        
    def _generate_equations_code(self, equations: List[str], is_nonlinear: bool = False) -> str:
        """Generate code for equation computation as a list eq = [ ... ]
        Args:
            equations: List of equation strings
            is_nonlinear: Whether these are nonlinear terms to be added to existing equations
        Returns:
            Generated equation computation code
        """
        eq_exprs = []
        diff_pattern = r"diff\((\w+),(\w+)(?:,(\d+))?\)"
        for eq in equations:
            eq_code = eq
            def replace_diff(match):
                var, dim, order = match.groups()
                order = int(order) if order else 1
                if order == 1:
                    return f"d{var}_{dim}"
                else:
                    return f"d{var}_{dim * order}"
            eq_code = re.sub(diff_pattern, replace_diff, eq_code)
            eq_exprs.append(eq_code)
            
        if is_nonlinear:
            # For nonlinear terms, add them to existing equations
            code_lines = ["# Add nonlinear terms"]
            for i, expr in enumerate(eq_exprs):
                code_lines.append(f"eq[{i}] = eq[{i}] + {expr}")
        else:
            # For linear terms, create the initial equations list
            code_lines = ["# Compute equations as a list"]
            code_lines.append(f"eq = [" + ", ".join(eq_exprs) + "]")
            
        return "\n".join(code_lines)
        
    def _generate_loss_code(self, num_equations: int) -> str:
        """Generate code for computing the PDE loss using eq[i] in a sum
        Args:
            num_equations: Number of equations
        Returns:
            Generated loss computation code
        """
        if num_equations == 1:
            return "pde_loss = torch.mean((eq[0] - source[0]) ** 2)"
        else:
            return f"pde_loss = torch.mean(sum((eq[i] - source[i]) ** 2 for i in range({num_equations})))"
        
    def update_code(self, equations: List[str], nonlinear_equations: Optional[List[str]] = None) -> None:
        """Update the physics loss code in the model file
        
        Args:
            equations: List of linear equation strings
            nonlinear_equations: Optional list of nonlinear equation strings
        """
        # Generate derivatives code
        derivatives_code = self._generate_derivatives_code(equations + (nonlinear_equations or []))
        
        # Generate equations code
        equations_code = self._generate_equations_code(equations)
        
        # Combine all equations
        if nonlinear_equations:
            nonlinear_code = self._generate_equations_code(nonlinear_equations, is_nonlinear=True)
            equations_code += "\n\n" + nonlinear_code
            
        # Generate loss code - use len(equations) since nonlinear terms are added to existing equations
        loss_code = self._generate_loss_code(len(equations))
        
        # Backup current file before making changes
        backup_path = self.model_path + ".backup"
        try:
            with open(self.model_path, "r") as f:
                current_content = f.read()
            with open(backup_path, "w") as f:
                f.write(current_content)
        except Exception as e:
            print(f"Warning: Failed to create backup file: {e}")
        
        # Read the model file
        with open(self.model_path, "r") as f:
            lines = f.readlines()
            
        # Find the section to replace
        start_marker = "# auto code begin"
        end_marker = "# auto code end"
        
        new_lines = []
        skip_mode = False
        found_section = False
        
        for line in lines:
            # Remove any pde_loss line globally
            if "pde_loss = torch.mean" in line:
                continue
            if start_marker in line:
                new_lines.append(line)  # Keep start marker line
                # Add appropriate indentation (8 spaces for function body)
                indented_code = "\n".join("        " + code_line if code_line.strip() else "" for code_line in (derivatives_code + "\n\n" + equations_code + "\n\n" + loss_code).strip().split("\n"))
                new_lines.append(indented_code + "\n\n")
                skip_mode = True
                found_section = True
            elif end_marker in line:
                skip_mode = False
                new_lines.append(line)  # Keep end marker line
            elif not skip_mode:
                new_lines.append(line)
        
        if not found_section:
            raise ValueError(f"Code segment to replace not found in file {self.model_path}")
            
        with open(self.model_path, "w") as f:
            f.writelines(new_lines)
            
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
        config_path: Path to config.json, default None uses preset path
    """
    generator = LossCodeGenerator(spatial_vars, vars_list, const_list, model_path)
    generator.update_code(linear_equations, nonlinear_equations)
    
    # Update config file and exit
    if case_dir:
      config_path = os.path.join(case_dir, "config.json")
      update_config_auto_code(config_path, False)
      print("Auto code completed, please check the net.py file, restart the program")
      sys.exit(0) 