import re
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import hashlib
import pickle


class CachedOperatorFactory:
    """高效缓存版本的算子工厂 - 避免重复生成"""

    def __init__(self, all_derivatives: Dict, constants: Dict = None):
        self.all_derivatives = all_derivatives
        self.constants = constants or {}
        
        # 全局缓存 - 避免重复创建相同的算子
        self._function_cache = {}
        self._compiled_cache = {}
        
        # 预编译常用表达式模式
        self._math_patterns = self._precompile_math_patterns()
        
        # 预处理变量映射
        self._var_mappings_cache = {}

    def _precompile_math_patterns(self):
        """预编译常用的数学函数替换模式"""
        return {
            "sin": re.compile(r"(?<!np\.)sin\b"),
            "cos": re.compile(r"(?<!np\.)cos\b"),
            "tan": re.compile(r"(?<!np\.)tan\b"),
            "exp": re.compile(r"(?<!np\.)exp\b"),
            "log": re.compile(r"(?<!np\.)log\b"),
            "sqrt": re.compile(r"(?<!np\.)sqrt\b")
        }

    def get_cached_operator(self, operator_terms: List[Dict], operator_name: str, 
                          is_nonlinear: bool = False) -> Optional[Callable]:
        """获取缓存的算子函数，如果不存在则创建并缓存"""
        
        # 生成唯一的缓存键
        cache_key = self._generate_cache_key(operator_terms, operator_name, is_nonlinear)
        
        if cache_key in self._function_cache:
            return self._function_cache[cache_key]
        
        # 创建新函数并缓存
        if not operator_terms:
            return None
            
        if is_nonlinear:
            func = self._create_optimized_nonlinear_function(operator_terms, operator_name)
        else:
            func = self._create_optimized_linear_function(operator_terms, operator_name)
            
        self._function_cache[cache_key] = func
        return func

    def _generate_cache_key(self, operator_terms: List[Dict], operator_name: str, 
                          is_nonlinear: bool) -> str:
        """生成唯一的缓存键"""
        key_data = {
            'terms': operator_terms,
            'name': operator_name,
            'nonlinear': is_nonlinear,
            'derivatives': self.all_derivatives,
            'constants': self.constants
        }
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(key_str).hexdigest()

    def _create_optimized_nonlinear_function(self, operator_terms: List[Dict], 
                                           operator_name: str) -> Callable:
        """创建优化的非线性函数 - 最小化运行时开销"""
        
        # 预编译所有项
        compiled_terms = []
        for term in operator_terms:
            compiled_term = self._precompile_term(term, operator_name, is_nonlinear=True)
            compiled_terms.append(compiled_term)

        def fast_nonlinear_function(features=None, u=None, coeffs=None, segment_idx=None):
            results = []
            n_points = u.shape[0] if u is not None else features[0].shape[0]

            for compiled_term in compiled_terms:
                if compiled_term is None:
                    results.append(np.zeros(n_points))
                    continue

                # 使用预编译的局部变量构建器
                local_vars = compiled_term['local_vars_builder'](features, u, coeffs, segment_idx)
                
                try:
                    # 直接使用预编译的可执行对象
                    result = compiled_term['executable'](local_vars)
                    results.append(result)
                except:
                    results.append(np.zeros(n_points))

            # F算子优化输出
            if operator_name == "F":
                ne = len(results)
                output = np.empty((n_points, ne), dtype=np.float64)
                for i, result in enumerate(results):
                    output[:, i] = result.flatten()[:n_points]
                return output
            
            return results

        # 添加元数据
        fast_nonlinear_function.operator_name = operator_name
        fast_nonlinear_function.terms = operator_terms
        fast_nonlinear_function.is_nonlinear = True
        fast_nonlinear_function.compiled_terms = compiled_terms

        return fast_nonlinear_function

    def _create_optimized_linear_function(self, operator_terms: List[Dict], 
                                        operator_name: str) -> Callable:
        """创建优化的线性函数"""
        
        compiled_terms = []
        for term in operator_terms:
            compiled_term = self._precompile_term(term, operator_name, is_nonlinear=False)
            compiled_terms.append(compiled_term)

        def fast_linear_function(features):
            n_points = features[0].shape[0]
            dgN = features[0].shape[1]
            ne = len(compiled_terms)
            
            # 预分配结果矩阵
            result_matrix = np.zeros((ne, n_points, ne * dgN), dtype=np.float64)

            for eq_idx, compiled_term in enumerate(compiled_terms):
                if compiled_term is None:
                    continue

                local_vars = compiled_term['local_vars_builder'](features, None, None, None)
                
                try:
                    linear_result = compiled_term['executable'](local_vars)
                    
                    # 优化的矩阵填充
                    if linear_result.shape == (n_points, dgN):
                        start_col = eq_idx * dgN
                        result_matrix[eq_idx, :, start_col:start_col + dgN] = linear_result
                    else:
                        result_matrix[eq_idx, :, eq_idx * dgN] = linear_result.flatten()
                except:
                    continue

            return result_matrix

        fast_linear_function.operator_name = operator_name
        fast_linear_function.terms = operator_terms
        fast_linear_function.is_nonlinear = False

        return fast_linear_function

    def _precompile_term(self, term: Dict, operator_name: str, is_nonlinear: bool) -> Dict:
        """预编译单个算子项 - 一次性处理所有可预处理的内容"""
        
        derivative_indices = term["derivative_indices"]
        symbolic_expr = term["symbolic_expr"]

        # 预处理表达式
        expr = symbolic_expr
        
        # 替换常数（预处理）
        for const_name, const_value in self.constants.items():
            expr = expr.replace(const_name, str(const_value))

        # 快速数学函数替换（使用预编译的正则）
        for func_name, pattern in self._math_patterns.items():
            expr = pattern.sub(f"np.{func_name}", expr)

        # 预处理变量映射
        var_mappings = self._get_cached_var_mappings(derivative_indices, is_nonlinear)

        try:
            # 预编译表达式
            compiled_expr = compile(expr, "<string>", "eval")
            
            # 创建优化的局部变量构建器
            if is_nonlinear:
                local_vars_builder = self._create_nonlinear_vars_builder(var_mappings)
            else:
                local_vars_builder = self._create_linear_vars_builder(var_mappings)
            
            # 创建可执行函数（避免重复eval调用）
            def executable(local_vars):
                local_vars["np"] = np
                return eval(compiled_expr, {"__builtins__": {}}, local_vars)
            
            return {
                'compiled_expr': compiled_expr,
                'expr_str': expr,
                'var_mappings': var_mappings,
                'local_vars_builder': local_vars_builder,
                'executable': executable
            }
            
        except:
            return None

    def _get_cached_var_mappings(self, derivative_indices: List[int], is_nonlinear: bool) -> List:
        """获取缓存的变量映射"""
        cache_key = (tuple(derivative_indices), is_nonlinear)
        
        if cache_key not in self._var_mappings_cache:
            var_mappings = []
            for deriv_idx in derivative_indices:
                for name, (v_idx, d_idx) in self.all_derivatives.items():
                    if d_idx == deriv_idx:
                        if is_nonlinear:
                            var_mappings.append((name, v_idx, deriv_idx))
                        else:
                            var_mappings.append((name, deriv_idx))
            self._var_mappings_cache[cache_key] = var_mappings
            
        return self._var_mappings_cache[cache_key]

    def _create_nonlinear_vars_builder(self, var_mappings: List) -> Callable:
        """创建非线性变量构建器"""
        def build_vars(features, u, coeffs, segment_idx):
            local_vars = {}
            
            # 检查是否只有零阶导数
            only_zeroth = all(deriv_idx == 0 for _, _, deriv_idx in var_mappings)
            
            if only_zeroth and u is not None:
                # 直接使用u值的快速路径
                for var_name, var_idx, _ in var_mappings:
                    local_vars[var_name] = u[:, var_idx] if u.ndim == 2 and var_idx < u.shape[1] else u
            else:
                # 使用features@coeffs
                for var_name, var_idx, deriv_idx in var_mappings:
                    if coeffs is not None:
                        current_coeffs = (
                            coeffs[segment_idx, deriv_idx, :] if coeffs.ndim == 3 else
                            coeffs[deriv_idx, :] if coeffs.ndim == 2 else
                            coeffs[deriv_idx]
                        )
                        local_vars[var_name] = features[deriv_idx] @ current_coeffs
                    else:
                        local_vars[var_name] = features[deriv_idx]
            
            return local_vars
        
        return build_vars

    def _create_linear_vars_builder(self, var_mappings: List) -> Callable:
        """创建线性变量构建器"""
        def build_vars(features, u, coeffs, segment_idx):
            local_vars = {}
            for var_name, deriv_idx in var_mappings:
                local_vars[var_name] = features[deriv_idx]
            return local_vars
        
        return build_vars

    def create_all_operators(self, operator_terms_dict: Dict) -> Dict[str, Callable]:
        """批量创建所有算子函数 - 使用缓存"""
        operators = {}
        operator_types = {"L1": False, "L2": False, "N": True, "F": True}

        for op_name, is_nonlinear in operator_types.items():
            if op_name in operator_terms_dict and operator_terms_dict[op_name]:
                func = self.get_cached_operator(
                    operator_terms_dict[op_name], op_name, is_nonlinear
                )
                if func:
                    operators[f"{op_name}_func"] = func

        return operators

    def clear_cache(self):
        """清空缓存（如果需要）"""
        self._function_cache.clear()
        self._compiled_cache.clear()
        self._var_mappings_cache.clear()

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'function_cache_size': len(self._function_cache),
            'compiled_cache_size': len(self._compiled_cache),
            'var_mappings_cache_size': len(self._var_mappings_cache)
        }


def create_operator_factory(
    all_derivatives: Dict, constants: Dict = None, optimized: bool = True
) -> CachedOperatorFactory:
    """创建缓存版本的算子工厂"""
    return CachedOperatorFactory(all_derivatives, constants)