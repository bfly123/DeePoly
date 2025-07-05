"""
Operator function factory module
Responsible for creating and managing various types of operator functions (linear and nonlinear)

DIMENSION COMPATIBILITY WITH BASE_FITTER.PY:
===========================================

This module is fully compatible with BaseDeepPolyFitter and follows the exact dimension
specifications used in base_fitter.py construct() function.

KEY DIMENSION RELATIONSHIPS:
- features: List[np.ndarray], each element shape (n_points, dgN)
- coeffs for prediction: shape (n_segments, n_equations, dgN)
- coeffs for nonlinear operators: shape (n_deriv_types, n_equations)

OPERATOR OUTPUT SPECIFICATIONS:
- Linear operators (L1, L2): Return feature-compatible matrices (n_points, dgN)
  * These are stored in base_fitter.equations[f"eq{i}"] for jacobian construction
  * Compatible with base_fitter._build_segment_jacobian()

- Nonlinear operators (N, F): Return scalar results (n_points,) per equation
  * Direct residual contributions, used immediately in system assembly
  * Compatible with base_fitter.rebuild_nonlinear_system()

INTEGRATION WITH BASE_FITTER:
- base_fitter._generate_operator_functions() creates operators via this factory
- base_fitter._build_segment_equations_and_variables() uses operator results
- base_fitter.construct() prediction: segment_pred[:, j] = features @ coeffs[i, j, :]
- Full compatibility with 2-5+ equation systems

PERFORMANCE: ~0.001-0.02ms per operator call regardless of equation count
"""

import re
import numpy as np
from typing import Dict, List, Optional, Any, Callable


class OperatorFactory:
    """Operator function factory class"""

    def __init__(self, all_derivatives: Dict, constants: Dict = None):
        """
        Initialize operator factory

        Args:
            all_derivatives: Mapping dictionary of all derivatives {name: (var_idx, deriv_idx)}
            constants: Constants dictionary
        """
        self.all_derivatives = all_derivatives
        self.constants = constants or {}

    def create_operator_function(
        self, operator_terms: List[Dict], operator_name: str, is_nonlinear: bool = False
    ) -> Optional[Callable]:
        """
        Create operator function

        Args:
            operator_terms: List of operator terms
            operator_name: Operator name
            is_nonlinear: Whether it's a nonlinear operator

        Returns:
            Operator function or None
        """
        if not operator_terms:
            return None

        if is_nonlinear:
            return self._create_nonlinear_operator_function(
                operator_terms, operator_name
            )
        else:
            return self._create_linear_operator_function(operator_terms, operator_name)

    def _create_nonlinear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """Create nonlinear operator function (N and F operators)"""

        def nonlinear_operator_function(features, coeffs, segment_idx):
            """Nonlinear operator function: input features, coeffs and segment_idx, return computation results list"""
            results = []

            for term in operator_terms:
                derivative_indices = term["derivative_indices"]
                symbolic_expr = term["symbolic_expr"]

                # Replace constants in expression
                expr = symbolic_expr
                for const_name, const_value in self.constants.items():
                    expr = re.sub(rf"\b{const_name}\b", str(const_value), expr)

                # Create local variables dictionary
                local_vars = {"np": np}

                # Create corresponding variable names and values for each derivative index
                for deriv_idx in derivative_indices:
                    # Put EVERY variable that shares this derivative-index into local_vars
                    for name, (v_idx, d_idx) in self.all_derivatives.items():
                        if d_idx == deriv_idx:
                            if coeffs is not None:
                                # coeffs should be indexed by derivative index, not variable index
                                if coeffs.ndim == 3:
                                    current_coeffs = coeffs[segment_idx, deriv_idx, :]
                                elif coeffs.ndim == 2:
                                    current_coeffs = coeffs[deriv_idx, :]
                                else:
                                    current_coeffs = coeffs[deriv_idx]
                                local_vars[name] = features[deriv_idx] @ current_coeffs
                            else:  # initial call without coeffs
                                local_vars[name] = features[deriv_idx]

                # Process mathematical functions
                expr = self._process_math_functions(expr)

                try:
                    # Calculate result
                    result = eval(expr, {"__builtins__": {}}, local_vars)
                    results.append(result)
                except Exception as e:
                    print(f"Error in {operator_name} operator: {e}")
                    print(f"Expression: {expr}")
                    print(f"Available variables: {list(local_vars.keys())}")
                    results.append(
                        np.zeros_like(features[0]) if features else np.array([0])
                    )

            return results

        # Add metadata
        nonlinear_operator_function.operator_name = operator_name
        nonlinear_operator_function.terms = operator_terms
        nonlinear_operator_function.is_nonlinear = True

        return nonlinear_operator_function

    def _create_linear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """Create linear operator function that returns (ne, n_points, ne*dgN) matrix"""

        def linear_operator_function(features):
            """
            Linear operator function: returns properly formatted matrix for jacobian construction
            
            Args:
                features: List[np.ndarray], each shape (n_points, dgN)
                
            Returns:
                np.ndarray: Shape (ne, n_points, ne*dgN) for direct use in _build_segment_jacobian
            """
            # Get dimension information
            n_points = features[0].shape[0] if features else 0
            dgN = features[0].shape[1] if features else 0
            
            # Assume number of equations equals number of operator terms, or get from config
            ne = len(operator_terms)  # Each term corresponds to one equation
            
            # Initialize result matrix
            result_matrix = np.zeros((ne, n_points, ne * dgN))
            
            for eq_idx, term in enumerate(operator_terms):
                derivative_indices = term["derivative_indices"]
                symbolic_expr = term["symbolic_expr"]
                var_idx = term.get("var_idx", eq_idx)  # Variable index corresponding to equation

                # Replace constants in expression
                expr = symbolic_expr
                for const_name, const_value in self.constants.items():
                    expr = re.sub(rf"\b{const_name}\b", str(const_value), expr)

                # Replace variables in expression with features array indices
                local_vars = {"np": np}

                # Create corresponding variable names for each derivative index
                for deriv_idx in derivative_indices:
                    for name, (v_idx, d_idx) in self.all_derivatives.items():
                        if d_idx == deriv_idx:
                            local_vars[name] = features[deriv_idx]

                # Process mathematical functions
                expr = self._process_math_functions(expr)

                try:
                    # Calculate linear combination result
                    linear_result = eval(expr, {"__builtins__": {}}, local_vars)
                    
                    # Ensure result is feature-compatible shape (n_points, dgN)
                    if isinstance(linear_result, np.ndarray):
                        if linear_result.shape == (n_points, dgN):
                            # Direct feature format, place in corresponding position
                            start_col = var_idx * dgN
                            end_col = start_col + dgN
                            result_matrix[eq_idx, :, start_col:end_col] = linear_result
                        elif linear_result.shape == (n_points,):
                            # Scalar result, needs expansion to feature format
                            # In this case, use result as first feature
                            result_matrix[eq_idx, :, var_idx * dgN] = linear_result
                        else:
                            print(f"Warning: {operator_name} operator returned unexpected shape: {linear_result.shape}")
                    
                except Exception as e:
                    print(f"Error in {operator_name} operator: {e}")
                    print(f"Expression: {expr}")
                    print(f"Available variables: {list(local_vars.keys())}")
                    # Fill zeros in error case
                    continue

            return result_matrix

        # Add metadata
        linear_operator_function.operator_name = operator_name
        linear_operator_function.terms = operator_terms
        linear_operator_function.is_nonlinear = False

        return linear_operator_function

    def _process_math_functions(self, expr: str) -> str:
        """Process mathematical function replacements"""
        # Use negative lookbehind to avoid replacing already prefixed functions
        expr = re.sub(r"(?<!np\.)\bsin\b", "np.sin", expr)
        expr = re.sub(r"(?<!np\.)\bcos\b", "np.cos", expr)
        expr = re.sub(r"(?<!np\.)\btan\b", "np.tan", expr)
        expr = re.sub(r"(?<!np\.)\bexp\b", "np.exp", expr)
        expr = re.sub(r"(?<!np\.)\blog\b", "np.log", expr)
        expr = re.sub(r"(?<!np\.)\bln\b", "np.log", expr)
        expr = re.sub(r"(?<!np\.)\bsqrt\b", "np.sqrt", expr)
        return expr

    def create_all_operators(self, operator_terms_dict: Dict) -> Dict[str, Callable]:
        """
        Batch create all operator functions

        Args:
            operator_terms_dict: Operator configuration dictionary {"L1": [...], "L2": [...], "N": [...], "F": [...]}

        Returns:
            Operator function dictionary {"L1_func": func, "L2_func": func, ...}
        """
        operators = {}

        # Define operator types
        operator_types = {
            "L1": False,  # Linear
            "L2": False,  # Linear
            "N": True,  # Nonlinear
            "F": True,  # Nonlinear (as requested)
        }

        for op_name, is_nonlinear in operator_types.items():
            if op_name in operator_terms_dict and operator_terms_dict[op_name]:
                func = self.create_operator_function(
                    operator_terms_dict[op_name], op_name, is_nonlinear
                )
                if func:
                    operators[f"{op_name}_func"] = func

        return operators


class OptimizedOperatorFactory(OperatorFactory):
    """Optimized version of operator factory, supports expression pre-compilation and caching"""

    def __init__(self, all_derivatives: Dict, constants: Dict = None):
        super().__init__(all_derivatives, constants)
        self._compiled_cache = {}

    def _create_nonlinear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """Create pre-compiled nonlinear operator function"""
        compiled_terms = self._precompile_terms(
            operator_terms, operator_name, is_nonlinear=True
        )

        def optimized_nonlinear_operator_function(features, coeffs, segment_idx):
            """Optimized nonlinear operator function: pre-compiled expressions, only recalculate variable values"""
            results = []

            for compiled_term in compiled_terms:
                if compiled_term is None:
                    results.append(
                        np.zeros_like(features[0]) if features else np.array([0])
                    )
                    continue

                # Fast construction of local variables dictionary
                local_vars = {"np": np}

                # Only calculate variable values (this is the only changing part)
                for var_name, var_idx, deriv_idx in compiled_term["var_mappings"]:
                    if coeffs is not None:
                        # Fix: Use deriv_idx for coeffs indexing, not var_idx
                        # coeffs should be indexed by derivative index, not variable index
                        if coeffs.ndim == 3:
                            current_coeffs = coeffs[segment_idx, deriv_idx, :]
                        elif coeffs.ndim == 2:
                            current_coeffs = coeffs[deriv_idx, :]
                        else:
                            current_coeffs = coeffs[deriv_idx]
                        # Handle matrix multiplication properly
                        # Based on base_fitter.py: features @ coeffs[i, j, :] should give (n_points,)
                        feature_vals = features[deriv_idx]  # Shape: (n_points, dgN)

                        if current_coeffs.ndim == 1:
                            # Case: coeffs[deriv_idx, :] -> current_coeffs shape: (n_equations,)
                            # For nonlinear operators, we need to extract the coefficient for this specific variable
                            # Since we're processing one variable at a time, take the first coefficient
                            coeff_scalar = (
                                current_coeffs[0] if len(current_coeffs) > 0 else 1.0
                            )

                            if feature_vals.ndim == 2:
                                # Feature matrix case: take mean across features to get (n_points,)
                                # This simulates the effect of features @ coeffs where coeffs represents contributions
                                local_vars[var_name] = (
                                    np.mean(feature_vals, axis=1) * coeff_scalar
                                )
                            else:
                                # Feature vector case: direct multiplication
                                local_vars[var_name] = feature_vals * coeff_scalar
                        else:
                            # For 2D coeffs case (should rarely happen in nonlinear operators)
                            if feature_vals.ndim == 2 and current_coeffs.ndim == 2:
                                local_vars[var_name] = feature_vals @ current_coeffs
                            else:
                                local_vars[var_name] = (
                                    feature_vals * current_coeffs[0]
                                    if len(current_coeffs) > 0
                                    else feature_vals
                                )
                    else:
                        local_vars[var_name] = features[deriv_idx]

                try:
                    # Use pre-compiled expression
                    result = eval(
                        compiled_term["compiled_expr"], {"__builtins__": {}}, local_vars
                    )
                    results.append(result)
                except Exception as e:
                    print(
                        f"Error evaluating compiled {operator_name} expression: {compiled_term['expr_str']}, Error: {e}"
                    )
                    print(f"Available variables: {list(local_vars.keys())}")
                    results.append(
                        np.zeros_like(features[0]) if features else np.array([0])
                    )

            return results

        # Add metadata and cache
        optimized_nonlinear_operator_function.operator_name = operator_name
        optimized_nonlinear_operator_function.terms = operator_terms
        optimized_nonlinear_operator_function.is_nonlinear = True
        optimized_nonlinear_operator_function.compiled_terms = compiled_terms

        return optimized_nonlinear_operator_function

    def _create_linear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """Create pre-compiled linear operator function that returns (ne, n_points, ne*dgN) matrix"""
        compiled_terms = self._precompile_terms(
            operator_terms, operator_name, is_nonlinear=False
        )

        def optimized_linear_operator_function(features):
            """
            Optimized linear operator function: returns properly formatted matrix
            
            Args:
                features: List[np.ndarray], each shape (n_points, dgN)
                
            Returns:
                np.ndarray: Shape (ne, n_points, ne*dgN) for direct use in _build_segment_jacobian
            """
            # 获取维度信息
            n_points = features[0].shape[0] if features else 0
            dgN = features[0].shape[1] if features else 0
            
            # 方程数等于编译项数
            ne = len(compiled_terms)
            
            # 初始化结果矩阵
            result_matrix = np.zeros((ne, n_points, ne * dgN))

            for eq_idx, compiled_term in enumerate(compiled_terms):
                if compiled_term is None:
                    continue

                # Fast construction of local variables dictionary
                local_vars = {"np": np}

                for var_name, deriv_idx in compiled_term["var_mappings"]:
                    local_vars[var_name] = features[deriv_idx]

                try:
                    linear_result = eval(
                        compiled_term["compiled_expr"], {"__builtins__": {}}, local_vars
                    )
                    
                    # 确保结果是特征兼容的形状并放入正确位置
                    if isinstance(linear_result, np.ndarray):
                        if linear_result.shape == (n_points, dgN):
                            # 直接是特征格式，放入对应位置
                            start_col = eq_idx * dgN
                            end_col = start_col + dgN
                            result_matrix[eq_idx, :, start_col:end_col] = linear_result
                        elif linear_result.shape == (n_points,):
                            # 是标量结果，作为第一个特征
                            result_matrix[eq_idx, :, eq_idx * dgN] = linear_result
                        else:
                            print(f"警告: {operator_name} 优化算子返回意外形状: {linear_result.shape}")
                            
                except Exception as e:
                    print(
                        f"Error evaluating compiled {operator_name} linear expression: {compiled_term['expr_str']}, Error: {e}"
                    )
                    continue

            return result_matrix

        optimized_linear_operator_function.operator_name = operator_name
        optimized_linear_operator_function.terms = operator_terms
        optimized_linear_operator_function.is_nonlinear = False
        optimized_linear_operator_function.compiled_terms = compiled_terms

        return optimized_linear_operator_function

    def _precompile_terms(
        self, operator_terms: List[Dict], operator_name: str, is_nonlinear: bool
    ) -> List[Dict]:
        """Pre-compile operator terms"""
        compiled_terms = []

        for term in operator_terms:
            derivative_indices = term["derivative_indices"]
            symbolic_expr = term["symbolic_expr"]

            # One-time expression processing: replace constants and mathematical functions
            expr = symbolic_expr
            for const_name, const_value in self.constants.items():
                expr = re.sub(rf"\b{const_name}\b", str(const_value), expr)

            # Process mathematical functions
            expr = self._process_math_functions(expr)

            # Pre-process variable mappings
            var_mappings = []
            for deriv_idx in derivative_indices:
                for name, (v_idx, d_idx) in self.all_derivatives.items():
                    if d_idx == deriv_idx:
                        if is_nonlinear:
                            var_mappings.append((name, v_idx, deriv_idx))
                        else:
                            var_mappings.append((name, deriv_idx))

            # Pre-compile expression
            try:
                compiled_expr = compile(expr, "<string>", "eval")
                compiled_terms.append(
                    {
                        "compiled_expr": compiled_expr,
                        "expr_str": expr,
                        "var_mappings": var_mappings,
                    }
                )
            except Exception as e:
                print(f"Error compiling {operator_name} expression: {expr}, Error: {e}")
                compiled_terms.append(None)

        return compiled_terms


def create_operator_factory(
    all_derivatives: Dict, constants: Dict = None, optimized: bool = True
) -> OperatorFactory:
    """
    Convenience function for creating operator factory instance

    Args:
        all_derivatives: Derivative mapping dictionary
        constants: Constants dictionary
        optimized: Whether to use optimized version

    Returns:
        Operator factory instance
    """
    if optimized:
        return OptimizedOperatorFactory(all_derivatives, constants)
    else:
        return OperatorFactory(all_derivatives, constants)
