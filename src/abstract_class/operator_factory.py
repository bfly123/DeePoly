import re
import numpy as np
from typing import Dict, List, Optional, Any, Callable


class OperatorFactory:
    """Optimized operator function factory - 简洁高效版本"""

    def __init__(self, all_derivatives: Dict, constants: Dict = None):
        self.all_derivatives = all_derivatives
        self.constants = constants or {}

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
        """创建非线性算子函数 - 简化版本"""

        def nonlinear_operator_function(features=None, u=None, coeffs=None, segment_idx=None):
            results = []

            for term in operator_terms:
                derivative_indices = term["derivative_indices"]
                symbolic_expr = term["symbolic_expr"]

                # 替换常数
                expr = symbolic_expr
                for const_name, const_value in self.constants.items():
                    expr = re.sub(rf"\b{const_name}\b", str(const_value), expr)

                # 创建局部变量字典
                local_vars = {"np": np}

                # 检查是否只包含零阶导数
                only_zeroth_derivatives = all(
                    any(d_idx == 0 for _, (_, d_idx) in self.all_derivatives.items() if d_idx == deriv_idx)
                    for deriv_idx in derivative_indices
                )
                
                if only_zeroth_derivatives and u is not None:
                    # 直接使用u值
                    for deriv_idx in derivative_indices:
                        for name, (v_idx, d_idx) in self.all_derivatives.items():
                            if d_idx == deriv_idx and d_idx == 0:
                                # 简化：假设u是标准的2D数组 (n_points, n_eqs)
                                local_vars[name] = u[:, v_idx] if u.ndim == 2 and v_idx < u.shape[1] else u
                else:
                    # 使用 features@coeffs
                    for deriv_idx in derivative_indices:
                        for name, (v_idx, d_idx) in self.all_derivatives.items():
                            if d_idx == deriv_idx:
                                if coeffs is not None:
                                    # 简化索引：直接使用标准形式
                                    current_coeffs = (
                                        coeffs[segment_idx, deriv_idx, :] if coeffs.ndim == 3 else
                                        coeffs[deriv_idx, :] if coeffs.ndim == 2 else
                                        coeffs[deriv_idx]
                                    )
                                    local_vars[name] = features[deriv_idx] @ current_coeffs
                                else:
                                    local_vars[name] = features[deriv_idx]

                # 处理数学函数
                expr = self._process_math_functions(expr)

                # 计算结果
                try:
                    result = eval(expr, {"__builtins__": {}}, local_vars)
                    results.append(result)
                except:
                    # 简化错误处理：直接用零数组
                    fallback_size = u.shape[0] if u is not None else features[0].shape[0]
                    results.append(np.zeros(fallback_size))

            # F算子标准化输出
            if operator_name == "F":
                n_points = u.shape[0]
                ne = len(results)
                output = np.zeros((n_points, ne))
                for i, result in enumerate(results):
                    output[:, i] = np.array(result).flatten()[:n_points]
                return output
            
            return results

        nonlinear_operator_function.operator_name = operator_name
        nonlinear_operator_function.terms = operator_terms
        nonlinear_operator_function.is_nonlinear = True

        return nonlinear_operator_function

    def _create_linear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """创建线性算子函数 - 简化版本"""

        def linear_operator_function(features):
            # 获取维度信息
            n_points = features[0].shape[0]
            dgN = features[0].shape[1]
            ne = len(operator_terms)
            
            # 初始化结果矩阵
            result_matrix = np.zeros((ne, n_points, ne * dgN))
            
            for eq_idx, term in enumerate(operator_terms):
                derivative_indices = term["derivative_indices"]
                symbolic_expr = term["symbolic_expr"]
                var_idx = term.get("var_idx", eq_idx)

                # 替换常数
                expr = symbolic_expr
                for const_name, const_value in self.constants.items():
                    expr = re.sub(rf"\b{const_name}\b", str(const_value), expr)

                # 创建变量字典
                local_vars = {"np": np}
                for deriv_idx in derivative_indices:
                    for name, (v_idx, d_idx) in self.all_derivatives.items():
                        if d_idx == deriv_idx:
                            local_vars[name] = features[deriv_idx]

                # 处理数学函数
                expr = self._process_math_functions(expr)

                # 计算线性组合结果
                try:
                    linear_result = eval(expr, {"__builtins__": {}}, local_vars)
                    
                    # 放入结果矩阵
                    if linear_result.shape == (n_points, dgN):
                        start_col = var_idx * dgN
                        end_col = start_col + dgN
                        result_matrix[eq_idx, :, start_col:end_col] = linear_result
                    else:
                        result_matrix[eq_idx, :, var_idx * dgN] = linear_result
                        
                except:
                    continue

            return result_matrix

        linear_operator_function.operator_name = operator_name
        linear_operator_function.terms = operator_terms
        linear_operator_function.is_nonlinear = False

        return linear_operator_function

    def _process_math_functions(self, expr: str) -> str:
        """处理数学函数替换"""
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

    def create_all_operators(self, operator_terms_dict: Dict) -> Dict[str, Callable]:
        """批量创建所有算子函数"""
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
    """预编译优化版本"""

    def __init__(self, all_derivatives: Dict, constants: Dict = None):
        super().__init__(all_derivatives, constants)
        self._compiled_cache = {}

    def _create_nonlinear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """创建预编译非线性算子函数"""
        compiled_terms = self._precompile_terms(operator_terms, operator_name, True)

        def optimized_nonlinear_operator_function(features=None, u=None, coeffs=None, segment_idx=None):
            results = []

            for compiled_term in compiled_terms:
                if compiled_term is None:
                    results.append(np.zeros(u.shape[0]))
                    continue

                # 快速构建局部变量
                local_vars = {"np": np}
                
                only_zeroth_derivatives = all(
                    deriv_idx == 0 for _, _, deriv_idx in compiled_term["var_mappings"]
                )

                if only_zeroth_derivatives and u is not None:
                    # 直接使用u值
                    for var_name, var_idx, deriv_idx in compiled_term["var_mappings"]:
                        if deriv_idx == 0:
                            local_vars[var_name] = u[:, var_idx] if u.ndim == 2 and var_idx < u.shape[1] else u
                else:
                    # 使用features@coeffs
                    for var_name, var_idx, deriv_idx in compiled_term["var_mappings"]:
                        if coeffs is not None:
                            current_coeffs = (
                                coeffs[segment_idx, deriv_idx, :] if coeffs.ndim == 3 else
                                coeffs[deriv_idx, :] if coeffs.ndim == 2 else
                                coeffs[deriv_idx]
                            )
                            
                            if current_coeffs.ndim == 1:
                                coeff_scalar = current_coeffs[0] if len(current_coeffs) > 0 else 1.0
                                local_vars[var_name] = np.mean(features[deriv_idx], axis=1) * coeff_scalar
                            else:
                                local_vars[var_name] = features[deriv_idx] @ current_coeffs
                        else:
                            local_vars[var_name] = features[deriv_idx]

                # 使用预编译表达式
                try:
                    result = eval(compiled_term["compiled_expr"], {"__builtins__": {}}, local_vars)
                    results.append(result)
                except:
                    results.append(np.zeros(u.shape[0]))

            # F算子标准化输出
            if operator_name == "F":
                n_points = u.shape[0]
                ne = len(results)
                output = np.zeros((n_points, ne))
                for i, result in enumerate(results):
                    output[:, i] = np.array(result).flatten()[:n_points]
                return output
                
            return results

        optimized_nonlinear_operator_function.operator_name = operator_name
        optimized_nonlinear_operator_function.terms = operator_terms
        optimized_nonlinear_operator_function.is_nonlinear = True

        return optimized_nonlinear_operator_function

    def _create_linear_operator_function(
        self, operator_terms: List[Dict], operator_name: str
    ) -> Callable:
        """创建预编译线性算子函数"""
        compiled_terms = self._precompile_terms(operator_terms, operator_name, False)

        def optimized_linear_operator_function(features):
            n_points = features[0].shape[0]
            dgN = features[0].shape[1]
            ne = len(compiled_terms)
            result_matrix = np.zeros((ne, n_points, ne * dgN))

            for eq_idx, compiled_term in enumerate(compiled_terms):
                if compiled_term is None:
                    continue

                # 快速构建局部变量
                local_vars = {"np": np}
                for var_name, deriv_idx in compiled_term["var_mappings"]:
                    local_vars[var_name] = features[deriv_idx]

                # 使用预编译表达式
                try:
                    linear_result = eval(compiled_term["compiled_expr"], {"__builtins__": {}}, local_vars)
                    
                    if linear_result.shape == (n_points, dgN):
                        start_col = eq_idx * dgN
                        end_col = start_col + dgN
                        result_matrix[eq_idx, :, start_col:end_col] = linear_result
                    else:
                        result_matrix[eq_idx, :, eq_idx * dgN] = linear_result
                except:
                    continue

            return result_matrix

        optimized_linear_operator_function.operator_name = operator_name
        optimized_linear_operator_function.terms = operator_terms
        optimized_linear_operator_function.is_nonlinear = False

        return optimized_linear_operator_function

    def _precompile_terms(self, operator_terms: List[Dict], operator_name: str, is_nonlinear: bool) -> List[Dict]:
        """预编译算子项"""
        compiled_terms = []

        for term in operator_terms:
            derivative_indices = term["derivative_indices"]
            symbolic_expr = term["symbolic_expr"]

            # 替换常数
            expr = symbolic_expr
            for const_name, const_value in self.constants.items():
                expr = re.sub(rf"\b{const_name}\b", str(const_value), expr)

            # 处理数学函数
            expr = self._process_math_functions(expr)

            # 预处理变量映射
            var_mappings = []
            for deriv_idx in derivative_indices:
                for name, (v_idx, d_idx) in self.all_derivatives.items():
                    if d_idx == deriv_idx:
                        if is_nonlinear:
                            var_mappings.append((name, v_idx, deriv_idx))
                        else:
                            var_mappings.append((name, deriv_idx))

            # 预编译表达式
            try:
                compiled_expr = compile(expr, "<string>", "eval")
                compiled_terms.append({
                    "compiled_expr": compiled_expr,
                    "expr_str": expr,
                    "var_mappings": var_mappings,
                })
            except:
                compiled_terms.append(None)

        return compiled_terms


def create_operator_factory(
    all_derivatives: Dict, constants: Dict = None, optimized: bool = True
) -> OperatorFactory:
    """创建算子工厂实例"""
    if optimized:
        return OptimizedOperatorFactory(all_derivatives, constants)
    else:
        return OperatorFactory(all_derivatives, constants)