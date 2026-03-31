import re
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from src.abstract_class.shape import ensure_points_eqs, safe_eq_col, broadcast_coeffs


class OperatorFactory:
    """Optimized operator function factory - 简洁高效Version"""

    def __init__(self, all_derivatives: Dict, constants: Dict = None):
        self.all_derivatives = all_derivatives
        self.constants = constants or {}
        self._derivatives_by_index = self._build_derivative_index_map()
        self.n_vars = max((var_idx for var_idx, _ in self.all_derivatives.values()), default=-1) + 1

    def _build_derivative_index_map(self) -> Dict[int, List[tuple]]:
        derivative_map: Dict[int, List[tuple]] = {}
        for name, (var_idx, deriv_idx) in self.all_derivatives.items():
            derivative_map.setdefault(deriv_idx, []).append((name, var_idx, deriv_idx))
        return derivative_map

    def create_operator_function(
        self, operator_terms: List[Dict], operator_name: str, is_nonlinear: bool = False
    ) -> Optional[Callable]:
        if not operator_terms:
            return None

        if is_nonlinear:
            return self._create_nonlinear_operator_function(operator_terms, operator_name)
        else:
            return self._create_linear_operator_function(operator_terms, operator_name)

    def _create_nonlinear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """CreateNonlinear operatorfunction - 简化Version"""

        def nonlinear_operator_function(features=None, u=None, coeffs=None, segment_idx=None):
            results = []

            for term in operator_terms:
                derivative_indices = term["derivative_indices"]
                expr = self._prepare_expression(term["symbolic_expr"])
                local_vars = self._build_nonlinear_local_vars(
                    derivative_indices=derivative_indices,
                    features=features,
                    u=u,
                    coeffs=coeffs,
                    segment_idx=segment_idx,
                )
                fallback_size = self._get_nonlinear_fallback_size(features, u)
                results.append(self._safe_eval(expr, local_vars, fallback_size))

            if operator_name == "F":
                return self._format_f_operator_output(results, features, u)
            
            return results

        nonlinear_operator_function.operator_name = operator_name
        nonlinear_operator_function.terms = operator_terms
        nonlinear_operator_function.is_nonlinear = True

        return nonlinear_operator_function

    def _create_linear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """CreateLinear operatorsfunction - 简化Version"""

        def linear_operator_function(features):
            # GetDimensionsinformation
            n_points = features[0].shape[0]
            dgN = features[0].shape[1]
            ne = len(operator_terms)

            # InitializeResultMatrix
            result_matrix = np.zeros((ne, n_points, self.n_vars * dgN))

            for eq_idx, term in enumerate(operator_terms):
                derivative_indices = term["derivative_indices"]
                expr = self._prepare_expression(term["symbolic_expr"])
                for var_idx in range(self.n_vars):
                    local_vars = self._build_linear_local_vars(
                        derivative_indices,
                        features,
                        active_var_idx=var_idx,
                    )
                    try:
                        linear_result = eval(expr, {"__builtins__": {}}, local_vars)
                        self._assign_linear_result(
                            result_matrix=result_matrix,
                            eq_idx=eq_idx,
                            var_idx=var_idx,
                            linear_result=linear_result,
                            n_points=n_points,
                            dgN=dgN,
                        )
                    except:
                        continue

            return result_matrix

        linear_operator_function.operator_name = operator_name
        linear_operator_function.terms = operator_terms
        linear_operator_function.is_nonlinear = False

        return linear_operator_function

    def _process_math_functions(self, expr: str) -> str:
        """ProcessMathematicalfunction替换"""
        replacements = {
            r"(?<!np\.)sin": "np.sin",
            r"(?<!np\.)cos": "np.cos", 
            r"(?<!np\.)tan": "np.tan",
            r"(?<!np\.)exp": "np.exp",
            r"(?<!np\.)log": "np.log",
            r"(?<!np\.)sqrt": "np.sqrt"
        }
        
        for pattern, replacement in replacements.items():
            expr = re.sub(pattern, replacement, expr)
        
        return expr

    def _prepare_expression(self, symbolic_expr: str) -> str:
        expr = symbolic_expr
        for const_name, const_value in self.constants.items():
            expr = re.sub(rf"\b{const_name}\b", str(const_value), expr)
        return self._process_math_functions(expr)

    def _iter_derivative_entries(self, deriv_idx: int) -> List[tuple]:
        return self._derivatives_by_index.get(deriv_idx, [])

    def _all_zeroth_derivatives(self, derivative_indices: List[int]) -> bool:
        return all(all(entry[2] == 0 for entry in self._iter_derivative_entries(deriv_idx)) for deriv_idx in derivative_indices)

    def _get_nonlinear_fallback_size(self, features=None, u=None) -> int:
        if u is not None:
            return u.shape[0]
        return features[0].shape[0]

    def _build_nonlinear_local_vars(self, derivative_indices, features=None, u=None, coeffs=None, segment_idx=None):
        local_vars = {"np": np}
        if self._all_zeroth_derivatives(derivative_indices) and u is not None:
            self._populate_zeroth_order_vars(local_vars, derivative_indices, u)
            return local_vars

        self._populate_feature_vars(local_vars, derivative_indices, features, coeffs, segment_idx)
        return local_vars

    def _populate_zeroth_order_vars(self, local_vars: Dict, derivative_indices: List[int], u) -> None:
        u_std = ensure_points_eqs(u, "u")
        for deriv_idx in derivative_indices:
            for name, var_idx, mapped_deriv_idx in self._iter_derivative_entries(deriv_idx):
                if mapped_deriv_idx == 0:
                    local_vars[name] = safe_eq_col(
                        u_std,
                        min(var_idx, u_std.shape[1] - 1),
                        name,
                    ).flatten()

    def _populate_feature_vars(self, local_vars: Dict, derivative_indices: List[int], features, coeffs=None, segment_idx=None) -> None:
        for deriv_idx in derivative_indices:
            for name, _, _ in self._iter_derivative_entries(deriv_idx):
                if coeffs is not None:
                    local_vars[name] = self._project_feature_with_coeffs(features, coeffs, deriv_idx, segment_idx)
                else:
                    local_vars[name] = features[deriv_idx]

    def _project_feature_with_coeffs(self, features, coeffs, deriv_idx: int, segment_idx=None):
        coeffs_standardized = np.atleast_3d(coeffs)
        seg_idx = min(segment_idx if segment_idx is not None else 0, coeffs_standardized.shape[0] - 1)
        current_coeffs = coeffs_standardized[seg_idx, deriv_idx, :]
        return features[deriv_idx] @ current_coeffs

    def _safe_eval(self, expr: str, local_vars: Dict, fallback_size: int):
        try:
            return eval(expr, {"__builtins__": {}}, local_vars)
        except:
            return np.zeros(fallback_size)

    def _format_f_operator_output(self, results: List, features=None, u=None) -> np.ndarray:
        n_points = self._get_nonlinear_fallback_size(features, u)
        ne = len(results)
        output = np.zeros((n_points, ne))
        for i, result in enumerate(results):
            output[:, i] = np.array(result).flatten()[:n_points]
        return output

    def _build_linear_local_vars(self, derivative_indices: List[int], features, active_var_idx: int) -> Dict:
        local_vars = {"np": np}
        for deriv_idx in derivative_indices:
            for name, var_idx, _ in self._iter_derivative_entries(deriv_idx):
                local_vars[name] = features[deriv_idx] if var_idx == active_var_idx else np.zeros_like(features[deriv_idx])
        return local_vars

    def _assign_linear_result(self, result_matrix: np.ndarray, eq_idx: int, var_idx: int, linear_result, n_points: int, dgN: int) -> None:
        if np.isscalar(linear_result):
            return
        if linear_result.shape == (n_points, dgN):
            start_col = var_idx * dgN
            end_col = start_col + dgN
            result_matrix[eq_idx, :, start_col:end_col] += linear_result
        else:
            result_matrix[eq_idx, :, var_idx * dgN] += linear_result

    def create_all_operators(self, operator_terms_dict: Dict) -> Dict[str, Callable]:
        """BatchCreateAllOperatorsfunction"""
        operators = {}
        operator_types = {"L1": False, "L2": False, "N": True, "F": True}

        for op_name, is_nonlinear in operator_types.items():
            if op_name in operator_terms_dict and operator_terms_dict[op_name]:
                func = self.create_operator_function(
                    operator_terms_dict[op_name], op_name, is_nonlinear
                )
                if func:
                    operators[f"{op_name}_func"] = func

        return operators


class OptimizedOperatorFactory(OperatorFactory):
    """预CompilationOptimizeVersion"""

    def __init__(self, all_derivatives: Dict, constants: Dict = None):
        super().__init__(all_derivatives, constants)
        self._compiled_cache = {}

    def _create_nonlinear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """Create预CompilationNonlinear operatorfunction"""
        compiled_terms = self._precompile_terms(operator_terms, operator_name, True)

        def optimized_nonlinear_operator_function(features=None, u=None, coeffs=None, segment_idx=None):
            results = []

            for compiled_term in compiled_terms:
                if compiled_term is None:
                    results.append(np.zeros(self._get_nonlinear_fallback_size(features, u)))
                    continue

                local_vars = self._build_compiled_nonlinear_local_vars(
                    compiled_term["var_mappings"],
                    features=features,
                    u=u,
                    coeffs=coeffs,
                    segment_idx=segment_idx,
                )
                try:
                    result = eval(compiled_term["compiled_expr"], {"__builtins__": {}}, local_vars)
                    results.append(result)
                except:
                    results.append(np.zeros(self._get_nonlinear_fallback_size(features, u)))

            if operator_name == "F":
                return self._format_f_operator_output(results, features, u)
                
            return results

        optimized_nonlinear_operator_function.operator_name = operator_name
        optimized_nonlinear_operator_function.terms = operator_terms
        optimized_nonlinear_operator_function.is_nonlinear = True

        return optimized_nonlinear_operator_function

    def _create_linear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """Create预CompilationLinear operatorsfunction"""
        compiled_terms = self._precompile_terms(operator_terms, operator_name, False)

        def optimized_linear_operator_function(features):
            n_points = features[0].shape[0]
            dgN = features[0].shape[1]
            ne = len(compiled_terms)
            result_matrix = np.zeros((ne, n_points, self.n_vars * dgN))

            for eq_idx, compiled_term in enumerate(compiled_terms):
                if compiled_term is None:
                    continue

                for var_idx in range(self.n_vars):
                    local_vars = self._build_compiled_linear_local_vars(
                        compiled_term["var_mappings"],
                        features,
                        active_var_idx=var_idx,
                    )
                    try:
                        linear_result = eval(compiled_term["compiled_expr"], {"__builtins__": {}}, local_vars)
                        self._assign_linear_result(
                            result_matrix=result_matrix,
                            eq_idx=eq_idx,
                            var_idx=var_idx,
                            linear_result=linear_result,
                            n_points=n_points,
                            dgN=dgN,
                        )
                    except:
                        continue

            return result_matrix

        optimized_linear_operator_function.operator_name = operator_name
        optimized_linear_operator_function.terms = operator_terms
        optimized_linear_operator_function.is_nonlinear = False

        return optimized_linear_operator_function

    def _precompile_terms(self, operator_terms: List[Dict], operator_name: str, is_nonlinear: bool) -> List[Dict]:
        """预CompilationOperatorsItem"""
        compiled_terms = []

        for term in operator_terms:
            compiled_terms.append(self._compile_term(term, is_nonlinear))

        return compiled_terms

    def _build_var_mappings(self, derivative_indices: List[int], is_nonlinear: bool) -> List[tuple]:
        var_mappings = []
        for deriv_idx in derivative_indices:
            for name, var_idx, mapped_deriv_idx in self._iter_derivative_entries(deriv_idx):
                var_mappings.append((name, var_idx, mapped_deriv_idx))
        return var_mappings

    def _compile_term(self, term: Dict, is_nonlinear: bool) -> Optional[Dict]:
        expr = self._prepare_expression(term["symbolic_expr"])
        var_mappings = self._build_var_mappings(term["derivative_indices"], is_nonlinear)
        try:
            return {
                "compiled_expr": compile(expr, "<string>", "eval"),
                "expr_str": expr,
                "var_mappings": var_mappings,
            }
        except:
            return None

    def _build_compiled_nonlinear_local_vars(self, var_mappings, features=None, u=None, coeffs=None, segment_idx=None):
        local_vars = {"np": np}
        only_zeroth_derivatives = all(deriv_idx == 0 for _, _, deriv_idx in var_mappings)
        if only_zeroth_derivatives and u is not None:
            u_std = ensure_points_eqs(u, "u")
            for var_name, var_idx, deriv_idx in var_mappings:
                if deriv_idx == 0:
                    local_vars[var_name] = safe_eq_col(
                        u_std,
                        min(var_idx, u_std.shape[1] - 1),
                        var_name,
                    ).flatten()
            return local_vars

        for var_name, _, deriv_idx in var_mappings:
            local_vars[var_name] = self._project_compiled_feature(features, coeffs, deriv_idx, segment_idx)
        return local_vars

    def _project_compiled_feature(self, features, coeffs, deriv_idx: int, segment_idx=None):
        if coeffs is None:
            return features[deriv_idx]

        coeffs_std = broadcast_coeffs(
            coeffs,
            (1, 1, coeffs.shape[-1]) if coeffs.ndim == 3 else (1, coeffs.shape[-1]),
            "coeffs",
        )
        if coeffs_std.ndim == 3:
            seg_idx = segment_idx if segment_idx is not None else 0
            current_coeffs = coeffs_std[seg_idx, deriv_idx, :]
        else:
            current_coeffs = coeffs_std[deriv_idx, :]

        if current_coeffs.ndim == 1:
            coeff_scalar = current_coeffs[0] if len(current_coeffs) > 0 else 1.0
            return np.mean(features[deriv_idx], axis=1) * coeff_scalar
        return features[deriv_idx] @ current_coeffs

    def _build_compiled_linear_local_vars(self, var_mappings, features, active_var_idx: int) -> Dict:
        local_vars = {"np": np}
        for var_name, var_idx, deriv_idx in var_mappings:
            local_vars[var_name] = (
                features[deriv_idx] if var_idx == active_var_idx else np.zeros_like(features[deriv_idx])
            )
        return local_vars


def create_operator_factory(
    all_derivatives: Dict, constants: Dict = None, optimized: bool = True
) -> OperatorFactory:
    """CreateOperator factoryinstance"""
    if optimized:
        return OptimizedOperatorFactory(all_derivatives, constants)
    else:
        return OperatorFactory(all_derivatives, constants)
