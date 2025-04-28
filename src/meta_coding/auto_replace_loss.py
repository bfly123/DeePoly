import os
import sys
import functools
from typing import Tuple, List, Dict, Any, Optional
import sympy as sp
import numpy as np
import shutil
from datetime import datetime
import logging
from abstract_class.base_net import BaseNet


class LossProcessor:
    def __init__(self, dimensions: List[str], vars_list: List[str], const_list: List[Dict[str, float]] = None):
        """
        Initialize loss function processor
        
        Args:
            dimensions: List of dimensions (e.g., ['x'] for 1D, ['x', 'y'] for 2D)
            vars_list: List of variables (e.g., ['u', 'v', 'p'])
            const_list: List of constant dictionaries (e.g., [{'Re': 100}])
        """
        self.dimensions = dimensions
        self.vars_list = vars_list
        self.const_list = const_list or []

        # Create symbolic variables
        self.vars = {dim: sp.Symbol(dim) for dim in dimensions}

        # Create basic variables
        self.var_symbols = {var: sp.Symbol(var) for var in vars_list}

        # Create constants symbols
        self.const_symbols = {}
        for const_dict in self.const_list:
            for const_name, const_value in const_dict.items():
                self.const_symbols[const_name] = sp.Symbol(const_name)
        
        # Create derivative symbols
        self.derivatives = {}
        for var in vars_list:
            for dim in dimensions:
                self.derivatives[f"{var}_{dim}"] = sp.Symbol(f"{var}_{dim}")
                
    def parse_equation(self, eq_str: str) -> sp.Expr:
        """
        Parse equation string to sympy expression
        
        Args:
            eq_str: Equation string
            
        Returns:
            Parsed sympy expression
            
        Raises:
            ValueError: Equation parsing error
        """
        if eq_str == "0":
            return sp.Integer(0)

        # Replace second-order derivative expressions - supporting both formats: diff(diff(u,x),y) and diff(u,x,2)
        for var in self.vars_list:
            # Replace diff(u,x,2) format
            for dim in self.dimensions:
                eq_str = eq_str.replace(f"diff({var},{dim},2)", f"{var}_{dim}{dim}")
            
            # Replace diff(diff(u,x),y) format
            for dim1 in self.dimensions:
                for dim2 in self.dimensions:
                    eq_str = eq_str.replace(f"diff(diff({var},{dim1}),{dim2})", f"{var}_{dim1}{dim2}")
        
        # Replace first-order derivative expressions
        for var in self.vars_list:
            for dim in self.dimensions:
                eq_str = eq_str.replace(f"diff({var},{dim})", f"{var}_{dim}")

        # Convert equation string to expression
        if '=' in eq_str:
            lhs, rhs = eq_str.split('=')
            eq_str = f"({lhs}) - ({rhs})"

        try:
            expr = sp.sympify(eq_str)
            return expr
        except Exception as e:
            raise ValueError(f"Equation parsing error: {str(e)}, Original equation: {eq_str}")

    def _find_used_derivatives(self, equations: Tuple[str, ...]) -> Dict[str, List[str]]:
        """
        Analyze derivatives used in equations
        
        Args:
            equations: System of equations
            
        Returns:
            List of derivatives needed for each variable
        """
        used_derivatives = {var: set() for var in self.vars_list}
        
        for eq in equations:
            if eq == "0":
                continue
                
            # Preprocess equation string to handle diff(u,x,2) format
            processed_eq = eq
            for var in self.vars_list:
                for dim in self.dimensions:
                    processed_eq = processed_eq.replace(f"diff({var},{dim},2)", f"diff(diff({var},{dim}),{dim})")
            
            expr = self.parse_equation(processed_eq)
            
            # Check derivatives
            for var in self.vars_list:
                for dim in self.dimensions:
                    deriv_sym = self.derivatives[f"{var}_{dim}"]
                    if expr.has(deriv_sym):
                        used_derivatives[var].add(dim)
                        
                        # Check if second-order derivatives are needed
                        for dim2 in self.dimensions:
                            second_deriv = f"{var}_{dim}{dim2}"
                            if second_deriv in str(expr):
                                used_derivatives[var].add(f"{dim}{dim2}")
        
        # Convert sets to lists
        return {var: list(derivs) for var, derivs in used_derivatives.items()}

    def generate_derivatives_code(self, equations: Tuple[str, ...]) -> str:
        """
        Generate PyTorch code for computing derivatives
        
        Args:
            equations: System of equations
            
        Returns:
            Generated derivatives computation code
        """
        used_derivatives = self._find_used_derivatives(equations)
        code_lines = []
        
        # Add constants from const_list - 修正访问格式为param[0]["constant_name"]
        for const_dict in self.const_list:
            for const_name, const_value in const_dict.items():
                code_lines.append(f"# Get {const_name} from parameters")
                code_lines.append(f"{const_name} = param[0][\"{const_name}\"]")
        
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
                    code_lines.append(f"d{var}_{dim} = BaseNet.gradients({var}, x_train)[0][..., {dim_idx}]")
        
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
                        code_lines.append(f"d{var}_{dim} = BaseNet.gradients({first_deriv}, x_train)[0][..., {dim2_idx}]")
        
        return "\n".join(code_lines)

    def generate_equations_code(self, equations: Tuple[str, ...]) -> str:
        """
        Generate PyTorch code for equation computation
        
        Args:
            equations: System of equations
            
        Returns:
            Generated equation computation code
        """
        code_lines = []
        code_lines.append("# N-S equation terms")
        
        for i, eq in enumerate(equations):
            if eq == "0":
                code_lines.append(f"eq{i} = torch.zeros_like({self.vars_list[0]})")
                continue
            
            # Create a deep copy of the equation for processing
            processed_eq = eq
            
            # Process equation for symbolic computation
            for var in self.vars_list:
                # First handle second-order derivatives with direct format
                for dim in self.dimensions:
                    processed_eq = processed_eq.replace(f"diff({var},{dim},2)", f"d2{var}_d{dim}2")
                
                # Then handle nested second-order derivatives
                for dim1 in self.dimensions:
                    for dim2 in self.dimensions:
                        processed_eq = processed_eq.replace(f"diff(diff({var},{dim1}),{dim2})", f"d2{var}_d{dim1}d{dim2}")
                        
                # Finally handle first-order derivatives
                for dim in self.dimensions:
                    processed_eq = processed_eq.replace(f"diff({var},{dim})", f"d{var}_d{dim}")
            
            # Handle division by Re
            processed_eq = processed_eq.replace("/Re", " / Re")
            
            # Format equation directly for PyTorch
            # Convert sympy-like variable names to PyTorch variable names
            for var in self.vars_list:
                # Convert second-order derivative names
                for dim in self.dimensions:
                    processed_eq = processed_eq.replace(f"d2{var}_d{dim}2", f"d{var}_{dim}{dim}")
                    
                for dim1 in self.dimensions:
                    for dim2 in self.dimensions:
                        if dim1 != dim2:  # Skip duplicate for clarity
                            processed_eq = processed_eq.replace(f"d2{var}_d{dim1}d{dim2}", f"d{var}_{dim1}{dim2}")
                
                # Convert first-order derivative names
                for dim in self.dimensions:
                    processed_eq = processed_eq.replace(f"d{var}_d{dim}", f"d{var}_{dim}")
                
            code_lines.append(f"eq{i} = {processed_eq}")
        
        return "\n".join(code_lines)

    def generate_loss_code(self, equations: Tuple[str, ...], output_path: str) -> None:
        """
        Generate complete loss function code
        
        Args:
            equations: System of equations
            output_path: Output file path
            
        Raises:
            Exception: Code generation error
        """
        try:
            # Generate derivative computation code
            derivatives_code = self.generate_derivatives_code(equations)
            
            # Generate equation code
            equations_code = self.generate_equations_code(equations)
            
            # Combine complete code
            code = f"""
{derivatives_code}

{equations_code}
"""
            # Write to file
            with open(output_path, "w") as f:
                f.write(code)

            print(f"Loss function code generated to: {output_path}")

        except Exception as e:
            print(f"Error generating loss function code: {str(e)}")
            raise


class LossCodeGenerator:
    """Loss function code generator class"""
    def __init__(self, dimensions: List[str], vars_list: List[str], const_list: List[Dict[str, float]] = None, model_path: Optional[str] = None):
        """
        Initialize loss function code generator
        
        Args:
            dimensions: List of dimensions (e.g., ['x'] for 1D, ['x', 'y'] for 2D)
            vars_list: List of variables (e.g., ['u', 'v', 'p'])
            const_list: List of constant dictionaries (e.g., [{'Re': 100}])
            model_path: Path to net.py, default None uses preset path
        """
        self.dimensions = dimensions
        self.vars_list = vars_list
        self.const_list = const_list or []
        self.processor = LossProcessor(dimensions, vars_list, const_list)
        self.model_path = model_path or "NewFramework/NS_stable/models/net.py"

    def update_code(self, linear_equations: List[str], nonlinear_equations: List[str] = None) -> None:
        """
        Update code in physics_loss function
        
        Args:
            linear_equations: Linear parts of the equations
            nonlinear_equations: Nonlinear parts of the equations (optional)
            
        Raises:
            FileNotFoundError: Target file not found
            ValueError: Code segment to replace not found
            Exception: Other update error
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"File not found: {self.model_path}")

            # Create backup
            backup_path = self._backup_file()
            
            # Combine equations if nonlinear_equations is provided
            combined_equations = []
            if nonlinear_equations:
                for i, lin_eq in enumerate(linear_equations):
                    if i < len(nonlinear_equations) and nonlinear_equations[i] != "0":
                        if lin_eq != "0":
                            combined_equations.append(f"{lin_eq} + {nonlinear_equations[i]}")
                        else:
                            combined_equations.append(nonlinear_equations[i])
                    else:
                        combined_equations.append(lin_eq)
            else:
                combined_equations = linear_equations
            
            # Generate and update code
            self._generate_and_update_code(tuple(combined_equations), backup_path)
            
        except Exception as e:
            print(f"Error updating loss function code: {str(e)}")
            raise

    def _backup_file(self) -> str:
        """
        Create file backup
        
        Returns:
            Backup file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.model_path}.{timestamp}.bak"
        shutil.copy2(self.model_path, backup_path)
        print(f"Backup created: {backup_path}")
        return backup_path

    def _generate_and_update_code(self, equations: Tuple[str, ...], backup_path: str) -> None:
        """
        Generate and update code
        
        Args:
            equations: System of equations
            backup_path: Backup file path
            
        Raises:
            Exception: Code generation or update error
        """
        # Generate temporary code file
        temp_file = os.path.join(os.path.dirname(__file__), "temp_generated_loss.txt")
        self.processor.generate_loss_code(equations, temp_file)

        try:
            # Read generated code
            with open(temp_file, "r") as f:
                generated_code = f.read()

            # Update original file
            self._update_file_content(generated_code)
            
            # Clean up temporary file
            os.remove(temp_file)
            print(f"Successfully updated {self.model_path} loss function")
            
        except Exception as e:
            print("Restoring backup...")
            shutil.copy2(backup_path, self.model_path)
            print("Backup restored")
            raise

    def _update_file_content(self, generated_code: str) -> None:
        """
        Update file content
        
        Args:
            generated_code: Generated code
            
        Raises:
            ValueError: Code segment to replace not found
        """
        start_marker = "# auto code begin"
        end_marker = "# auto code end"
        
        with open(self.model_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        skip_mode = False
        found_section = False

        for line in lines:
            if start_marker in line:
                new_lines.append(line)  # Keep start marker line
                
                # Add appropriate indentation
                indented_code = "\n".join("    " + code_line for code_line in generated_code.strip().split("\n"))
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
    """
    Convenient function to update code in physics_loss function
    
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


# Example code
if __name__ == "__main__":
    # Test run
    dimensions = ["x", "y"]
    vars_list = ["u", "v", "p"]
    const_list = [{"Re": 100}]
    
    # N-S equation (incompressible fluid)
    linear_equations = [
        "diff(u,x) + diff(v,y)",  # Continuity equation
        "diff(p,x) - diff(u,x,2)/Re - diff(u,y,2)/Re",  # Linear x-momentum terms
        "diff(p,y) - diff(v,x,2)/Re - diff(v,y,2)/Re",  # Linear y-momentum terms
    ]
    
    nonlinear_equations = [
        "0",  # No nonlinear terms in continuity
        "u*diff(u,x) + v*diff(u,y)",  # Nonlinear x-momentum terms
        "u*diff(v,x) + v*diff(v,y)",  # Nonlinear y-momentum terms
    ]
    
    print("Testing generating loss function code...")
    update_physics_loss_code(linear_equations, nonlinear_equations, vars_list, dimensions, const_list)