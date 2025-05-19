import re
from typing import List, Dict, Tuple, Optional
from src.abstract_class.base_net import BaseNet

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
                
            # Add derivative dimension repeated by order
            derivatives[var].append(dim * order)
            
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
        
    def _generate_equations_code(self, equations: List[str]) -> str:
        """Generate code for equation computation
        
        Args:
            equations: List of equation strings
            
        Returns:
            Generated equation computation code
        """
        code_lines = []
        code_lines.append("# Compute equations")
        
        for i, eq in enumerate(equations):
            # Replace diff() terms with their corresponding derivative variables
            eq_code = eq
            diff_pattern = r"diff\((\w+),(\w+)(?:,(\d+))?\)"
            
            def replace_diff(match):
                var, dim, order = match.groups()
                order = int(order) if order else 1
                if order == 1:
                    return f"d{var}_{dim}"
                else:
                    return f"d{var}_{dim * order}"
                    
            eq_code = re.sub(diff_pattern, replace_diff, eq_code)
            code_lines.append(f"eq{i} = {eq_code}")
            
        return "\n".join(code_lines)
        
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
            nonlinear_code = self._generate_equations_code(nonlinear_equations)
            equations_code += "\n\n# Add nonlinear terms\n" + nonlinear_code
            
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
            if start_marker in line:
                new_lines.append(line)  # Keep start marker line
                
                # Add appropriate indentation
                indented_code = "\n".join("    " + code_line for code_line in (derivatives_code + "\n\n" + equations_code).strip().split("\n"))
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
) -> None:
    """Convenient function to update code in physics_loss function
    
    Args:
        linear_equations: Linear parts of the equations
        nonlinear_equations: Nonlinear parts of the equations (optional)
        vars_list: List of variables
        spatial_vars: List of spatial dimensions
        const_list: List of constant dictionaries
        model_path: Path to net.py, default None uses preset path
    """
    generator = LossCodeGenerator(spatial_vars, vars_list, const_list, model_path)
    generator.update_code(linear_equations, nonlinear_equations) 