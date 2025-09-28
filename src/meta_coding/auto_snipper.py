"""
Multi-Operator Equation Parser - Simplified Version
Focus on: 1) Derivatives list 2) Max derivative orders 3) Operator term decomposition
"""

import re
from typing import List, Dict, Any, Optional


class OperatorParser:
    """Multi-operator equation parser - simplified version"""

    def __init__(
        self,
        variables: List[str],
        spatial_vars: List[str],
        constants: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize parser

        Args:
            variables: Variable list, e.g. ["u", "v", "p"]
            spatial_vars: Spatial variable list, e.g. ["x", "y", "z"]
            constants: Constants dictionary, e.g. {"Re": 100, "Pr": 0.7}
        """
        self.variables = variables
        self.spatial_vars = spatial_vars
        self.constants = constants or {}

    def extract_unique_derivatives(self, expressions: List[str]) -> List[List[int]]:
        """Extract unique derivative patterns, variable-independent"""
        unique_patterns = set()

        # Add 0-order derivatives (variables themselves)
        zero_order = tuple([0] * len(self.spatial_vars))
        unique_patterns.add(zero_order)

        # Extract derivative patterns from expressions
        for expr in expressions:
            matches = re.findall(r"diff\((\w+),(\w+)(?:,(\d+))?\)", expr)
            for var_name, spatial_var, order_str in matches:
                if var_name in self.variables and spatial_var in self.spatial_vars:
                    spatial_idx = self.spatial_vars.index(spatial_var)
                    order = int(order_str) if order_str else 1

                    # Create derivative pattern
                    pattern = [0] * len(self.spatial_vars)
                    pattern[spatial_idx] = order
                    unique_patterns.add(tuple(pattern))

        # Convert to list and sort
        derivatives = [list(pattern) for pattern in sorted(unique_patterns)]
        return derivatives

    def extract_all_derivatives_new(self, expressions: List[str], derivatives: List[List[int]]) -> Dict[str, List[int]]:
        """Re-extract all derivative terms based on derivatives list"""
        all_derivatives = {}

        # Create entries for each variable's derivative patterns
        for var_idx, var in enumerate(self.variables):
            for deriv_idx, deriv_pattern in enumerate(derivatives):
                # Create derivative name
                name = var.upper()
                if any(order > 0 for order in deriv_pattern):
                    for spatial_idx, order in enumerate(deriv_pattern):
                        if order > 0:
                            name += "_" + self.spatial_vars[spatial_idx] * order

                # Check if this derivative is used in expressions
                is_used = False
                
                # Check 0-order derivatives (variables themselves)
                if all(order == 0 for order in deriv_pattern):
                    for expr in expressions:
                        if re.search(rf"\b{var}\b(?!\s*[\,\)])", expr):
                            is_used = True
                            break
                else:
                    # Check higher-order derivatives
                    for expr in expressions:
                        # Build matching pattern
                        for spatial_idx, order in enumerate(deriv_pattern):
                            if order > 0:
                                spatial_var = self.spatial_vars[spatial_idx]
                                if order == 1:
                                    pattern = rf"diff\({var},{spatial_var}\)"
                                else:
                                    pattern = rf"diff\({var},{spatial_var},{order}\)"
                                if re.search(pattern, expr):
                                    is_used = True
                                    break
                        if is_used:
                            break

                # Only add derivatives that are actually used
                if is_used:
                    all_derivatives[name] = [var_idx, deriv_idx]

        return all_derivatives

    def extract_all_derivatives(self, expressions: List[str]) -> Dict[str, List[int]]:
        """Extract all derivative terms from expression strings"""
        derivatives = {}

        # Add 0-order derivatives (variables themselves)
        for i, var in enumerate(self.variables):
            name = var.upper()
            derivatives[name] = [i] + [0] * len(self.spatial_vars)

        # Extract derivatives from expressions
        for expr in expressions:
            # Find all diff patterns
            matches = re.findall(r"diff\((\w+),(\w+)(?:,(\d+))?\)", expr)

            for var_name, spatial_var, order_str in matches:
                if var_name in self.variables and spatial_var in self.spatial_vars:
                    var_idx = self.variables.index(var_name)
                    spatial_idx = self.spatial_vars.index(spatial_var)
                    order = int(order_str) if order_str else 1

                    # Create derivative name
                    name = var_name.upper()
                    if order > 0:
                        name += "_" + spatial_var * order

                    # Create order list
                    order_list = [0] * len(self.spatial_vars)
                    order_list[spatial_idx] = order

                    derivatives[name] = [var_idx] + order_list

            # Find standalone variables
            for var in self.variables:
                if re.search(rf"\b{var}\b(?!\s*[\,\)])", expr):
                    name = var.upper()
                    var_idx = self.variables.index(var)
                    derivatives[name] = [var_idx] + [0] * len(self.spatial_vars)

        return derivatives

    def get_max_orders(self, expressions: List[str]) -> List[List[int]]:
        """Get maximum derivative orders for each variable as a list"""
        # Initialize max orders for each variable
        max_orders = [[0] * len(self.spatial_vars) for _ in self.variables]

        for expr in expressions:
            matches = re.findall(r"diff\((\w+),(\w+)(?:,(\d+))?\)", expr)
            for var_name, spatial_var, order_str in matches:
                if var_name in self.variables and spatial_var in self.spatial_vars:
                    var_idx = self.variables.index(var_name)  # GetvariableIndex
                    spatial_idx = self.spatial_vars.index(spatial_var)
                    order = int(order_str) if order_str else 1
                    max_orders[var_idx][spatial_idx] = max(
                        max_orders[var_idx][spatial_idx], order
                    )

        return max_orders

    def replace_with_names(self, expr: str) -> str:
        """Replace derivatives and variables with standardized names"""

        # Replace diff(var,x,n) with VAR_xxx
        def replace_diff(match):
            var_name = match.group(1)
            spatial_var = match.group(2)
            order = int(match.group(3)) if match.group(3) else 1
            name = var_name.upper()
            if order > 0:
                name += "_" + spatial_var * order
            return name

        processed = re.sub(r"diff\((\w+),(\w+)(?:,(\d+))?\)", replace_diff, expr)

        # Replace variables with uppercase
        for var in self.variables:
            processed = re.sub(rf"\b{var}\b", var.upper(), processed)

        return processed

    def decompose_terms(
        self, expr: str, derivatives: Dict[str, List[int]]
    ) -> Dict[str, Any]:
        """Decompose operator terms and identify derivative indices"""
        if not expr:
            return {"derivative_indices": [], "symbolic_expr": expr}

        # Replace with standardized names
        processed = self.replace_with_names(expr)

        # Find all derivative names in the processed expression and get their deriv_idx
        indices = set()
        for name in derivatives.keys():
            if re.search(rf"\b{name}\b", processed):
                # Use the deriv_idx from all_derivatives [var_idx, deriv_idx]
                var_idx, deriv_idx = derivatives[name]
                indices.add(deriv_idx)

        return {"derivative_indices": sorted(indices), "symbolic_expr": processed}

    def parse(self, operators: Dict[str, List[str]]) -> Dict[str, Any]:
        """Parse multiple operators"""
        # Collect all expressions
        all_expressions = []
        for equations in operators.values():
            all_expressions.extend(equations)

        # Extract unique derivative patterns
        derivatives = self.extract_unique_derivatives(all_expressions)
        
        # Re-extract all_derivatives based on derivatives list
        all_derivatives = self.extract_all_derivatives_new(all_expressions, derivatives)
        
        # Extract max orders (keep original functionality)
        max_orders = self.get_max_orders(all_expressions)

        # Decompose operator terms
        operator_terms = {}
        for op_name, equations in operators.items():
            operator_terms[op_name] = []
            for expr in equations:
                terms = self.decompose_terms(expr, all_derivatives)
                operator_terms[op_name].append(terms)

        return {
            "derivatives": derivatives,  # New: unique derivative patterns list
            "all_derivatives": all_derivatives,  # Modified: new format [variable_idx, derivatives_idx]
            "max_derivative_orders": max_orders,
            "operator_terms": operator_terms
        }


def parse_operators(
    operators: Dict[str, List[str]],
    variables: List[str],
    spatial_vars: List[str],
    constants: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Simple interface function"""
    parser = OperatorParser(variables, spatial_vars, constants)
    return parser.parse(operators)


# New: public function for use by other modules
def parse_equation_to_list(
    equations: List[str],
    eq_nonlinear: List[str],
    variables: List[str],
    spatial_vars: List[str],
    constants: List[str],
) -> tuple:
    """
    Parse equations and return lists for compatibility with existing code
    This function maintains backward compatibility with the existing interface
    """
    # Combine all equations
    all_equations = equations + eq_nonlinear
    
    # Create operators dict
    operators = {
        "linear": equations,
        "nonlinear": eq_nonlinear
    }
    
    # Convert constants list to dict (if needed)
    constants_dict = {const: 1.0 for const in constants} if constants else {}
    
    # Parse using the main function
    result = parse_operators(operators, variables, spatial_vars, constants_dict)
    
    # Extract results in the expected format
    eq_linear_list = []  # This would need more specific implementation based on your needs
    deriv_orders = []
    max_deriv_orders = result["max_derivative_orders"]
    eq_nonlinear_list = []  # This would need more specific implementation
    all_derivatives = result["derivatives"]
    
    return (
        eq_linear_list,
        deriv_orders, 
        max_deriv_orders,
        eq_nonlinear_list,
        all_derivatives
    )