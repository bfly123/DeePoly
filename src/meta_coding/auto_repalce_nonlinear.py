import os
import sys
import functools
from typing import Tuple, List, Dict, Any, Optional
import sympy as sp
import numpy as np
import shutil
from datetime import datetime
import logging


class EquationProcessor:
    def __init__(self, dimensions: List[str], vars_list: List[str]):
        """
        初始化方程处理器
        
        Args:
            dimensions: 维度列表 (例如: ['x'] 表示1D, ['x', 'y'] 表示2D)
            vars_list: 变量列表 (例如: ['u', 'v', 'p'])
        """
        self.dimensions = dimensions
        self.vars_list = vars_list

        # 创建符号变量
        self.vars = {dim: sp.Symbol(dim) for dim in dimensions}

        # 创建基本变量
        self.var_symbols = {var: sp.Symbol(var) for var in vars_list}

        # 创建导数符号
        self.derivatives = {}
        for var in vars_list:
            for dim in dimensions:
                self.derivatives[f"{var}_{dim}"] = sp.Symbol(f"{var}_{dim}")

    def parse_equation(self, eq_str: str) -> sp.Expr:
        """
        解析方程字符串为sympy表达式
        
        Args:
            eq_str: 方程字符串
            
        Returns:
            解析后的sympy表达式
            
        Raises:
            ValueError: 方程解析错误
        """
        if eq_str == "0":
            return sp.Integer(0)

        # 替换导数表达式
        for var in self.vars_list:
            for dim in self.dimensions:
                eq_str = eq_str.replace(f"diff({var},{dim})", f"{var}_{dim}")

        # 将方程字符串转换为表达式
        if '=' in eq_str:
            lhs, rhs = eq_str.split('=')
            eq_str = f"({lhs}) - ({rhs})"

        try:
            expr = sp.sympify(eq_str)
            return expr
        except Exception as e:
            raise ValueError(f"方程解析错误: {str(e)}, 原方程: {eq_str}")

    def _format_multiplication(self, factors: List[str]) -> str:
        """
        将乘法因子格式化为NumPy代码
        
        Args:
            factors: 乘法因子列表
            
        Returns:
            格式化后的NumPy乘法代码
        """
        if len(factors) == 2:
            return f"np.multiply({factors[0]}, {factors[1]})"
        else:
            return "functools.reduce(np.multiply, [" + ", ".join(factors) + "])"

    def generate_L_terms(self, equations: Tuple[str, ...]) -> str:
        """
        生成残差项代码
        
        Args:
            equations: 方程组
            
        Returns:
            生成的残差项代码
        """
        L_terms = []

        for i, eq in enumerate(equations):
            if eq == "0":
                L_terms.append(f"L[{i}] = np.zeros((n_points, 1))")
            else:
                expr = self.parse_equation(eq)
                terms = []
                for term in sp.Add.make_args(expr):
                    # 将项分解为乘法因子
                    factors = sp.Mul.make_args(term)
                    if len(factors) > 1:
                        # 如果有多个因子，使用reduce形式的multiply
                        factors_str = [str(f) for f in factors]
                        numpy_term = self._format_multiplication(factors_str)
                    else:
                        numpy_term = str(term)
                    terms.append(numpy_term)

                if terms:
                    L_terms.append(f"L[{i}] = " + " + ".join(terms))
                else:
                    L_terms.append(f"L[{i}] = np.zeros((n_points, 1))")

        return "\n".join(L_terms)

    def generate_dL_terms(self, equations: Tuple[str, ...]) -> str:
        """
        生成雅可比矩阵项代码
        
        Args:
            equations: 方程组
            
        Returns:
            生成的雅可比矩阵项代码
        """
        dL_terms = []

        for i, eq in enumerate(equations):
            expr = self.parse_equation(eq)
            if expr == 0:
                dL_terms.append(f"dL[{i}] = np.zeros((n_points, ne * self.dgN))")
            else:
                # 生成雅可比项
                jac_terms = []
                for var in self.vars_list:
                    # 对变量本身求导
                    var_sym = sp.Symbol(var)
                    var_deriv = sp.diff(expr, var_sym)
                    if var_deriv != 0:
                        # 处理变量的系数
                        coeff_str = str(var_deriv)
                        jac_terms.append(
                            f"np.diag(({coeff_str}).flatten()) @ {var.upper()}"
                        )

                    # 对导数项求导
                    for dim in self.dimensions:
                        deriv_sym = self.derivatives[f"{var}_{dim}"]
                        deriv_coeff = sp.diff(expr, deriv_sym)
                        if deriv_coeff != 0:
                            deriv_coeff_str = str(deriv_coeff)
                            jac_terms.append(
                                f"np.diag(({deriv_coeff_str}).flatten()) @ {var.upper()}_{dim}"
                            )

                if jac_terms:
                    dL_terms.append(f"dL[{i}] = " + " + ".join(jac_terms))
                else:
                    dL_terms.append(f"dL[{i}] = np.zeros((n_points, ne * self.dgN))")

        return "\n".join(dL_terms)

    def _find_used_variables(self, equations: Tuple[str, ...]) -> Dict[str, bool]:
        """
        分析方程中使用的变量和导数
        
        Args:
            equations: 方程组
            
        Returns:
            使用的变量和导数的字典
        """
        used_vars = {var: False for var in self.vars_list}
        used_derivatives = {f"{var}_{dim}": False for var in self.vars_list for dim in self.dimensions}
        
        for eq in equations:
            if eq == "0":
                continue
                
            expr = self.parse_equation(eq)
            
            # 检查变量
            for var in self.vars_list:
                var_sym = sp.Symbol(var)
                if expr.has(var_sym):
                    used_vars[var] = True
            
            # 检查导数
            for var in self.vars_list:
                for dim in self.dimensions:
                    deriv_sym = self.derivatives[f"{var}_{dim}"]
                    if expr.has(deriv_sym):
                        used_derivatives[f"{var}_{dim}"] = True
                        used_vars[var] = True  # 如果导数被使用，变量也被认为是使用的
        
        # 合并结果
        result = {**used_vars, **used_derivatives}
        return result

    def generate_variable_declarations(self, equations: Tuple[str, ...], vars_list: List[str], dimensions: List[str]) -> str:
        """
        生成变量声明代码，只包含方程中使用的变量
        
        Args:
            equations: 方程组
            vars_list: 变量列表
            dimensions: 维度列表
            
        Returns:
            生成的变量声明代码
        """
        used_vars = self._find_used_variables(equations)
        declarations = []
        
        # 生成主变量声明
        for var in vars_list:
            if used_vars.get(var, False):
                declarations.append(f"{var.upper()} = variables[\"{var.upper()}\"][segment_idx]")
        
        # 生成导数变量声明
        for var in vars_list:
            for dim in dimensions:
                deriv_name = f"{var}_{dim}"
                if used_vars.get(deriv_name, False):
                    declarations.append(
                        f"{var.upper()}_{dim} = variables[\"{var.upper()}_{dim}\"][segment_idx]"
                    )
        
        # 生成局部变量声明
        for var in vars_list:
            if used_vars.get(var, False):
                declarations.append(f"{var} = {var.upper()} @ x_slice")
                for dim in dimensions:
                    deriv_name = f"{var}_{dim}"
                    if used_vars.get(deriv_name, False):
                        declarations.append(f"{var}_{dim} = {var.upper()}_{dim} @ x_slice")
        
        return "\n".join(" " * 8 + line for line in declarations)

    def generate_code_for_hybrid_fitter(
        self, equations: Tuple[str, ...], output_path: str
    ) -> None:
        """
        生成完整的代码，包括变量声明
        
        Args:
            equations: 方程组
            output_path: 输出文件路径
            
        Raises:
            Exception: 代码生成错误
        """
        try:
            # 生成变量声明，传递方程信息
            var_declarations = self.generate_variable_declarations(equations, self.vars_list, self.dimensions)
            
            # 生成L和dL项
            L_code = self.generate_L_terms(equations)
            dL_code = self.generate_dL_terms(equations)

            # 添加缩进
            indent = " " * 8
            L_code = "\n".join(indent + line for line in L_code.split("\n"))
            dL_code = "\n".join(indent + line for line in dL_code.split("\n"))

            # 组合完整代码
            code = f"""#-----begin auto code-----
{var_declarations}

{L_code}
{dL_code}
#-----end auto code-----
"""
            # 写入文件
            with open(output_path, "w") as f:
                f.write(code)

            print(f"代码已生成到: {output_path}")

        except Exception as e:
            print(f"生成代码时出错: {str(e)}")
            raise


class NonlinearCodeGenerator:
    """非线性代码生成器类"""
    def __init__(self, dimensions: List[str], vars_list: List[str], hybrid_fitter_path: Optional[str] = None):
        """
        初始化代码生成器
        
        Args:
            dimensions: 维度列表 (例如: ['x'] 表示1D, ['x', 'y'] 表示2D)
            vars_list: 变量列表 (例如: ['u', 'v', 'p'])
            hybrid_fitter_path: hybrid_fitter.py的路径，默认为None则使用预设路径
        """
        self.dimensions = dimensions
        self.vars_list = vars_list
        self.processor = EquationProcessor(dimensions, vars_list)
        self.hybrid_fitter_path = hybrid_fitter_path or "NewFramework/NS_stable/models/hybrid_fitter.py"

    def update_code(self, equations: Tuple[str, ...]) -> None:
        """
        更新HybridFitter中的代码
        
        Args:
            equations: 方程组
            
        Raises:
            FileNotFoundError: 找不到目标文件
            ValueError: 未找到需要替换的代码段
            Exception: 其他更新错误
        """
        try:
            if not os.path.exists(self.hybrid_fitter_path):
                raise FileNotFoundError(f"找不到文件: {self.hybrid_fitter_path}")

            # 创建备份
            backup_path = self._backup_file()
            
            # 生成和更新代码
            self._generate_and_update_code(equations, backup_path)
            
        except Exception as e:
            print(f"更新代码时出错: {str(e)}")
            raise

    def _backup_file(self) -> str:
        """
        创建文件备份
        
        Returns:
            备份文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.hybrid_fitter_path}.{timestamp}.bak"
        shutil.copy2(self.hybrid_fitter_path, backup_path)
        print(f"已创建备份: {backup_path}")
        return backup_path

    def _generate_and_update_code(self, equations: Tuple[str, ...], backup_path: str) -> None:
        """
        生成并更新代码
        
        Args:
            equations: 方程组
            backup_path: 备份文件路径
            
        Raises:
            Exception: 代码生成或更新错误
        """
        # 生成临时代码文件
        temp_file = os.path.join(os.path.dirname(__file__), "temp_generated_code.txt")
        self.processor.generate_code_for_hybrid_fitter(equations, temp_file)

        try:
            # 读取生成的代码
            with open(temp_file, "r") as f:
                generated_code = f.read()

            # 更新原文件
            self._update_file_content(generated_code)
            
            # 清理临时文件
            os.remove(temp_file)
            print(f"已成功更新 {self.hybrid_fitter_path}")
            
        except Exception as e:
            print("正在恢复备份...")
            shutil.copy2(backup_path, self.hybrid_fitter_path)
            print("已恢复备份")
            raise

    def _update_file_content(self, generated_code: str) -> None:
        """
        更新文件内容
        
        Args:
            generated_code: 生成的代码
            
        Raises:
            ValueError: 未找到需要替换的代码段
        """
        start_marker = "#-----begin auto code-----"
        end_marker = "#-----end auto code-----"
        
        with open(self.hybrid_fitter_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        skip_mode = False
        found_section = False

        for line in lines:
            if start_marker in line:
                new_lines.append(generated_code + "\n")
                skip_mode = True
                found_section = True
            elif end_marker in line:
                skip_mode = False
                continue
            elif not skip_mode:
                new_lines.append(line)

        if not found_section:
            raise ValueError(f"在文件 {self.hybrid_fitter_path} 中未找到需要替换的代码段")

        with open(self.hybrid_fitter_path, "w") as f:
            f.writelines(new_lines)


def update_hybrid_fitter_code(
    equations: Tuple[str, ...],
    vars_list: List[str],
    dimensions: List[str],
    hybrid_fitter_path: Optional[str] = None,
) -> None:
    """
    更新HybridFitter类中的代码的便捷函数
    
    Args:
        equations: 方程组
        vars_list: 变量列表
        dimensions: 维度列表
        hybrid_fitter_path: hybrid_fitter.py的路径，默认为None则使用预设路径
    """
    generator = NonlinearCodeGenerator(dimensions, vars_list, hybrid_fitter_path)
    generator.update_code(equations)

# 测试代码已移至单独的测试文件中
