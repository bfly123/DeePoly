import os
import sys
import functools
from typing import Tuple, List, Dict, Any, Optional, Union
import sympy as sp
import numpy as np
import shutil
from datetime import datetime
import logging
import json
import re


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
                # 处理高阶导数
                for order in range(2, 6):  # 支持到5阶导数
                    old_pattern = f"diff({var},{dim},{order})"
                    new_pattern = f"{var}_" + dim * order
                    eq_str = eq_str.replace(old_pattern, new_pattern)

        # 将方程字符串转换为表达式
        if '=' in eq_str:
            lhs, rhs = eq_str.split('=')
            eq_str = f"({lhs}) - ({rhs})"

        try:
            expr = sp.sympify(eq_str)
            return expr
        except Exception as e:
            raise ValueError(f"方程解析错误: {str(e)}, 原方程: {eq_str}")

    def _find_used_derivatives(self, equations: Union[List[str], Dict[str, List[str]]]) -> Dict[str, int]:
        """
        分析方程中使用的导数及其最高阶数
        
        Args:
            equations: 方程组
            
        Returns:
            使用的导数及其最高阶数的字典
        """
        used_derivatives = {}
        
        # 统一处理格式
        eq_list = []
        if isinstance(equations, dict):
            for op_name, eq_items in equations.items():
                if isinstance(eq_items, list):
                    eq_list.extend(eq_items)
                else:
                    eq_list.append(eq_items)
        else:
            eq_list = equations

        for eq in eq_list:
            if eq == "0":
                continue
                
            # 分析各种导数模式
            for var in self.vars_list:
                # 检查变量本身
                if var in eq:
                    key = f"{var}"
                    used_derivatives[key] = max(used_derivatives.get(key, 0), 0)
                
                for dim in self.dimensions:
                    # 检查各阶导数
                    for order in range(1, 6):
                        if order == 1:
                            pattern = f"diff({var},{dim})"
                        else:
                            pattern = f"diff({var},{dim},{order})"
                        
                        if pattern in eq:
                            # 对于高阶导数，也需要所有低阶导数
                            for lower_order in range(1, order + 1):
                                if lower_order == 1:
                                    key = f"{var}_{dim}"
                                else:
                                    key = f"{var}_{dim}" + dim * (lower_order - 1)
                                used_derivatives[key] = max(used_derivatives.get(key, 0), lower_order)
        
        return used_derivatives

    def generate_pytorch_derivatives(self, equations: Union[List[str], Dict[str, List[str]]]) -> str:
        """
        生成PyTorch神经网络的导数计算代码
        
        Args:
            equations: 方程组
            
        Returns:
            生成的导数计算代码
        """
        used_derivatives = self._find_used_derivatives(equations)
        derivatives_code = []
        
        # 生成变量提取代码
        derivatives_code.append("        # Extract physical quantities from output")
        for i, var in enumerate(self.vars_list):
            if any(key.startswith(var) for key in used_derivatives.keys()):
                derivatives_code.append(f"        {var} = U[..., {i}]")
        
        # 生成一阶导数代码
        first_order_added = False
        for var in self.vars_list:
            for i, dim in enumerate(self.dimensions):
                deriv_key = f"{var}_{dim}"
                if deriv_key in used_derivatives:
                    if not first_order_added:
                        derivatives_code.append("")
                        derivatives_code.append("        # Calculate derivatives in each direction")
                        first_order_added = True
                    derivatives_code.append(f"        d{var}_{dim} = self.gradients({var}, x_train)[0][..., {i}]")
        
        # 生成高阶导数代码
        for order in range(2, 6):
            order_added = False
            for var in self.vars_list:
                for i, dim1 in enumerate(self.dimensions):
                    for j, dim2 in enumerate(self.dimensions):
                        if order == 2:
                            deriv_key = f"{var}_{dim1}{dim2}"
                            if deriv_key in used_derivatives:
                                if not order_added:
                                    derivatives_code.append("")
                                    derivatives_code.append(f"        # Calculate {order}nd-order derivatives")
                                    order_added = True
                                derivatives_code.append(f"        d{var}_{dim1}{dim2} = self.gradients(d{var}_{dim1}, x_train)[0][..., {j}]")
                        elif order > 2:
                            deriv_key = f"{var}_{dim1}" + dim2 * (order - 1)
                            if deriv_key in used_derivatives:
                                if not order_added:
                                    derivatives_code.append("")
                                    derivatives_code.append(f"        # Calculate {order}th-order derivatives")
                                    order_added = True
                                prev_deriv = f"d{var}_{dim1}" + dim2 * (order - 2)
                                derivatives_code.append(f"        d{var}_{dim1}{dim2 * (order - 1)} = self.gradients({prev_deriv}, x_train)[0][..., {j}]")
        
        return "\n".join(derivatives_code)



    def _convert_equation_to_pytorch(self, eq: str) -> str:
        """
        将方程字符串转换为PyTorch格式
        
        Args:
            eq: 方程字符串
            
        Returns:
            PyTorch格式的方程字符串
        """
        eq_pytorch = eq
        for var in self.vars_list:
            for dim in self.dimensions:
                # 处理高阶导数
                for order in range(5, 0, -1):  # 从高阶到低阶
                    if order == 1:
                        old_pattern = f"diff({var},{dim})"
                        new_pattern = f"d{var}_{dim}"
                    else:
                        old_pattern = f"diff({var},{dim},{order})"
                        new_pattern = f"d{var}_{dim}" + dim * (order - 1)
                    eq_pytorch = eq_pytorch.replace(old_pattern, new_pattern)
        
        return eq_pytorch

    def generate_unified_operators(self, config_dict: Dict) -> str:
        """
        生成统一的算子代码，适用于所有问题类型
        
        Args:
            config_dict: 配置字典
            
        Returns:
            生成的算子代码
        """
        operators_code = []
        
        # 获取eq字段
        eq_dict = config_dict.get("eq", {})
        
        # 处理不同的算子类型
        for op_name, eq_list in eq_dict.items():
            operators_code.append(f"        # {op_name} operators")
            
            if isinstance(eq_list, list):
                # 生成列表格式
                eq_terms = []
                for eq in eq_list:
                    eq_pytorch = self._convert_equation_to_pytorch(eq)
                    eq_terms.append(eq_pytorch)
                
                if len(eq_terms) == 1:
                    operators_code.append(f"        {op_name} = [{eq_terms[0]}]")
                else:
                    operators_code.append(f"        {op_name} = [")
                    for i, term in enumerate(eq_terms):
                        if i == len(eq_terms) - 1:
                            operators_code.append(f"            {term}")
                        else:
                            operators_code.append(f"            {term},")
                    operators_code.append("        ]")
            else:
                # 如果不是列表，当作单个方程处理
                eq_pytorch = self._convert_equation_to_pytorch(eq_list)
                operators_code.append(f"        {op_name} = [{eq_pytorch}]")
            
            if op_name != list(eq_dict.keys())[-1]:  # 不是最后一个算子，添加空行
                operators_code.append("")
        
        # 对于时间PDE，还需要处理额外的算子字段
        if config_dict.get("problem_type") == "time_pde":
            # 处理时间PDE特有的算子
            time_operators = ["eq_L1", "eq_L2", "f_L2", "N"]
            for op_type in time_operators:
                if op_type in config_dict and config_dict[op_type]:
                    if op_type == "f_L2":
                        operators_code.append("")
                        operators_code.append("        # f_L2 functions (nonlinear functions for L2)")
                        op_display_name = "f_L2"
                    elif op_type == "N":
                        operators_code.append("")
                        operators_code.append("        # N operators (fully nonlinear)")
                        op_display_name = "N"
                    elif op_type == "eq_L1":
                        operators_code.append("")
                        operators_code.append("        # Additional L1 operators (implicit linear)")
                        op_display_name = "L1_extra"
                    elif op_type == "eq_L2":
                        operators_code.append("")
                        operators_code.append("        # Additional L2 operators (semi-implicit linear)")
                        op_display_name = "L2_extra"
                    
                    # 生成列表格式
                    eq_terms = []
                    for eq in config_dict[op_type]:
                        eq_pytorch = self._convert_equation_to_pytorch(eq)
                        eq_terms.append(eq_pytorch)
                    
                    if len(eq_terms) == 1:
                        operators_code.append(f"        {op_display_name} = [{eq_terms[0]}]")
                    else:
                        operators_code.append(f"        {op_display_name} = [")
                        for i, term in enumerate(eq_terms):
                            if i == len(eq_terms) - 1:
                                operators_code.append(f"            {term}")
                            else:
                                operators_code.append(f"            {term},")
                        operators_code.append("        ]")
        
        return "\n".join(operators_code)

    def generate_code_for_pytorch_net(
        self, equations: Union[List[str], Dict[str, List[str]]], 
        output_path: str, 
        problem_type: str = "linear_pde",
        config_dict: Dict = None
    ) -> None:
        """
        生成PyTorch神经网络的完整代码，只包含导数和算子部分
        
        Args:
            equations: 方程组（现在主要用于导数分析）
            output_path: 输出文件路径
            problem_type: 问题类型
            config_dict: 配置字典
            
        Raises:
            Exception: 代码生成错误
        """
        try:
            # 收集所有方程用于导数分析
            all_equations = []
            
            # 从eq字段收集方程
            if config_dict and "eq" in config_dict:
                eq_dict = config_dict["eq"]
                for op_name, eq_list in eq_dict.items():
                    if isinstance(eq_list, list):
                        all_equations.extend(eq_list)
                    else:
                        all_equations.append(eq_list)
            
            # 对于时间PDE，还要从其他字段收集方程
            if problem_type == "time_pde" and config_dict:
                time_operators = ["eq_L1", "eq_L2", "f_L2", "N"]
                for op_type in time_operators:
                    if op_type in config_dict and config_dict[op_type]:
                        all_equations.extend(config_dict[op_type])
            
            # 生成导数计算代码
            derivatives_code = self.generate_pytorch_derivatives(all_equations)
            
            # 生成统一的算子代码
            operators_code = self.generate_unified_operators(config_dict) if config_dict else ""

            # 生成配置签名（用于后续一致性检查）
            config_signature = ""
            if config_dict:
                # 提取算子配置
                sig_dict = {}
                if "eq" in config_dict:  # 新格式
                    eq = config_dict["eq"]
                    sig_dict = {
                        "L1": eq.get("L1", []),
                        "L2": eq.get("L2", []),
                        "F": eq.get("F", []),
                        "N": eq.get("N", [])
                    }
                else:  # 兼容旧格式
                    sig_dict = {
                        "L1": config_dict.get("eq_L1", []),
                        "L2": config_dict.get("eq_L2", []),
                        "F": config_dict.get("f_L2", []),
                        "N": config_dict.get("N", [])
                    }

                import json
                sig_str = json.dumps(sig_dict, sort_keys=True)
                config_signature = f"# Config signature: {sig_str}\n"

            # 组合代码
            code = f"""# auto code begin
{config_signature}{derivatives_code}

{operators_code}

# auto code end"""
            
            # 写入文件
            with open(output_path, "w") as f:
                f.write(code)

            print(f"PyTorch神经网络代码已生成到: {output_path}")

        except Exception as e:
            print(f"生成PyTorch代码时出错: {str(e)}")
            raise


class AutoCodeGenerator:
    """自动代码生成器类"""
    def __init__(self, config_path: str):
        """
        初始化自动代码生成器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config_dict = self._load_config()
        self.problem_type = self.config_dict.get("problem_type", "linear_pde")
        self.dimensions = self.config_dict.get("spatial_vars", ["x", "y"])
        self.vars_list = self.config_dict.get("vars_list", ["u"])
        
        self.processor = EquationProcessor(self.dimensions, self.vars_list)

    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {str(e)}")

    def check_config_net_consistency(self, net_file_path: str) -> Tuple[bool, str]:
        """
        检查配置文件中的算子定义与net.py中生成代码的一致性

        Args:
            net_file_path: net.py文件路径

        Returns:
            (需要重新生成, 原因说明)
        """
        # 检查net.py是否存在
        if not os.path.exists(net_file_path):
            return True, "net.py文件不存在"

        # 读取net.py内容
        with open(net_file_path, 'r') as f:
            net_content = f.read()

        # 检查是否有auto code块
        if "# auto code begin" not in net_content or "# auto code end" not in net_content:
            return True, "net.py中没有auto code标记"

        # 提取auto code块
        begin_idx = net_content.find("# auto code begin")
        end_idx = net_content.find("# auto code end")
        auto_code_block = net_content[begin_idx:end_idx]

        # 检查auto code块是否为空
        lines = auto_code_block.split('\n')[1:-1]
        has_code = any(line.strip() and not line.strip().startswith('#') for line in lines)

        if not has_code:
            return True, "auto code块为空，需要生成代码"

        # 提取当前配置的算子签名
        config_signature = self._extract_config_signature()

        # 在auto code块中查找配置签名
        if "# Config signature:" in auto_code_block:
            # 提取保存的签名
            for line in lines:
                if "# Config signature:" in line:
                    saved_sig = line.split("# Config signature:")[1].strip()
                    try:
                        saved_dict = json.loads(saved_sig)
                        current_dict = json.loads(config_signature)

                        # 比较两个签名
                        if saved_dict == current_dict:
                            return False, "config与net.py代码一致"
                        else:
                            # 找出具体的差异
                            diff_msg = self._find_signature_diff(saved_dict, current_dict)
                            return True, f"config算子定义已更改: {diff_msg}"
                    except:
                        return True, "无法解析保存的签名"

        # 如果没有签名，通过更抽象的方式检查
        return self._abstract_consistency_check(auto_code_block, config_signature)

    def _extract_config_signature(self) -> str:
        """提取配置文件的算子签名"""
        sig_dict = {}

        if "eq" in self.config_dict:  # 新格式
            eq = self.config_dict["eq"]
            sig_dict = {
                "L1": eq.get("L1", []),
                "L2": eq.get("L2", []),
                "F": eq.get("F", []),
                "N": eq.get("N", [])
            }
        else:  # 兼容旧格式
            sig_dict = {
                "L1": self.config_dict.get("eq_L1", []),
                "L2": self.config_dict.get("eq_L2", []),
                "F": self.config_dict.get("f_L2", []),
                "N": self.config_dict.get("N", [])
            }

        return json.dumps(sig_dict, sort_keys=True)

    def _find_signature_diff(self, saved_dict: Dict, current_dict: Dict) -> str:
        """找出两个签名字典的差异"""
        diffs = []

        for key in ["L1", "L2", "F", "N"]:
            saved = saved_dict.get(key, [])
            current = current_dict.get(key, [])

            if saved != current:
                if not saved and current:
                    diffs.append(f"{key}添加了算子")
                elif saved and not current:
                    diffs.append(f"{key}删除了算子")
                else:
                    diffs.append(f"{key}算子已修改")

        return ", ".join(diffs) if diffs else "未知差异"

    def _abstract_consistency_check(self, auto_code_block: str, config_signature: str) -> Tuple[bool, str]:
        """
        抽象的一致性检查，不依赖签名
        通过分析代码结构和算子模式进行判断
        """
        try:
            config_dict = json.loads(config_signature)

            # 检查每种算子类型
            for op_type, ops in config_dict.items():
                if not ops:
                    continue

                # 根据算子类型确定代码中的标记
                if op_type == "L1":
                    pattern = r"L1\s*=\s*\[(.*?)\]"
                elif op_type == "L2":
                    pattern = r"L2\s*=\s*\[(.*?)\]"
                elif op_type == "F":
                    pattern = r"F\s*=\s*\[(.*?)\]"
                elif op_type == "N":
                    pattern = r"N\s*=\s*\[(.*?)\]"
                else:
                    continue

                # 在代码中查找对应的算子定义
                match = re.search(pattern, auto_code_block, re.DOTALL)

                if not match and ops:
                    return True, f"{op_type}算子在代码中未找到"

                if match:
                    code_content = match.group(1).strip()

                    # 简单检查：算子数量是否一致
                    # 通过计算逗号数量来估计算子个数
                    expected_count = len(ops)
                    if code_content:
                        # 简单估计：如果有内容，至少有一个算子
                        code_has_content = bool(code_content and not code_content.isspace())
                        if expected_count > 0 and not code_has_content:
                            return True, f"{op_type}算子定义不匹配"
                    elif expected_count > 0:
                        return True, f"{op_type}算子在代码中为空"

            # 默认认为一致
            return False, "通过抽象检查，代码与配置基本一致"

        except Exception as e:
            # 出错时保守处理，认为需要重新生成
            return True, f"一致性检查出错: {str(e)}"

    def _get_net_file_path(self) -> str:
        """根据问题类型确定net.py文件路径"""
        case_dir = os.path.dirname(self.config_path)
        
        if self.problem_type == "linear_pde":
            return "src/problem_solvers/linear_pde_solver/core/net.py"
        elif self.problem_type == "time_pde":
            return "src/problem_solvers/time_pde_solver/core/net.py"
        elif self.problem_type == "func_fitting":
            return "src/problem_solvers/func_fitting_solver/core/net.py"
        else:
            raise ValueError(f"不支持的问题类型: {self.problem_type}")

    def _get_equations(self) -> Union[List[str], Dict[str, List[str]]]:
        """获取方程组用于导数分析"""
        all_equations = []
        
        # 从eq字段收集方程
        if "eq" in self.config_dict and self.config_dict["eq"]:
            eq_dict = self.config_dict["eq"]
            for op_name, eq_list in eq_dict.items():
                if isinstance(eq_list, list):
                    all_equations.extend(eq_list)
                else:
                    all_equations.append(eq_list)
        
        # 对于时间PDE，还要从其他字段收集方程
        if self.problem_type == "time_pde":
            time_operators = ["eq_L1", "eq_L2", "f_L2", "N"]
            for op_type in time_operators:
                if op_type in self.config_dict and self.config_dict[op_type]:
                    all_equations.extend(self.config_dict[op_type])
        
        return all_equations

    def update_code(self) -> None:
        """
        更新神经网络文件中的代码
        
        Raises:
            FileNotFoundError: 找不到目标文件
            ValueError: 未找到需要替换的代码段
            Exception: 其他更新错误
        """
        try:
            net_file_path = self._get_net_file_path()
            
            if not os.path.exists(net_file_path):
                raise FileNotFoundError(f"找不到文件: {net_file_path}")

            # 不再创建备份
            # backup_path = self._backup_file(net_file_path)

            # 生成和更新代码
            self._generate_and_update_code(net_file_path, None)
            
        except Exception as e:
            print(f"更新代码时出错: {str(e)}")
            raise

    def _backup_file(self, file_path: str) -> str:
        """
        创建文件备份
        
        Args:
            file_path: 要备份的文件路径
            
        Returns:
            备份文件路径
        """
        # 不再创建备份文件
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # backup_path = f"{file_path}.{timestamp}.bak"
        # shutil.copy2(file_path, backup_path)
        print(f"跳过备份创建 (已禁用)")
        return file_path  # 返回原文件路径

    def _generate_and_update_code(self, net_file_path: str, backup_path: str = None) -> None:
        """
        生成并更新代码

        Args:
            net_file_path: 目标文件路径
            backup_path: 备份文件路径（已废弃，保留参数以兼容）
            
        Raises:
            Exception: 代码生成或更新错误
        """
        # 生成临时代码文件
        temp_file = os.path.join(os.path.dirname(__file__), "temp_pytorch_code.txt")
        equations = self._get_equations()
        
        self.processor.generate_code_for_pytorch_net(
            equations, 
            temp_file, 
            self.problem_type,
            self.config_dict
        )

        try:
            # 读取生成的代码
            with open(temp_file, "r") as f:
                generated_code = f.read()

            # 更新原文件
            self._update_file_content(net_file_path, generated_code)
            
            # 清理临时文件
            os.remove(temp_file)
            print(f"已成功更新 {net_file_path}")
            
        except Exception as e:
            # 不再恢复备份，因为没有创建备份
            print(f"更新文件时出错: {str(e)}")
            print("注意：未创建备份文件，原文件可能已被修改")
            raise

    def _update_file_content(self, file_path: str, generated_code: str) -> None:
        """
        更新文件内容
        
        Args:
            file_path: 文件路径
            generated_code: 生成的代码
            
        Raises:
            ValueError: 未找到需要替换的代码段
        """
        start_marker = "# auto code begin"
        end_marker = "# auto code end"
        
        with open(file_path, "r") as f:
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
            raise ValueError(f"在文件 {file_path} 中未找到需要替换的代码段")

        with open(file_path, "w") as f:
            f.writelines(new_lines)


def update_physics_loss_from_config(config_path: str) -> None:
    """
    从配置文件自动更新physics loss代码的便捷函数
    
    Args:
        config_path: 配置文件路径
    """
    generator = AutoCodeGenerator(config_path)
    generator.update_code()


# 保留原有函数以保持向后兼容性
def update_pytorch_net_code(
    equations: Tuple[str, ...],
    vars_list: List[str],
    dimensions: List[str],
    net_file_path: str,
) -> None:
    """
    更新PyTorch神经网络类中的代码的便捷函数（保持向后兼容性）
    
    Args:
        equations: 方程组
        vars_list: 变量列表
        dimensions: 维度列表
        net_file_path: net.py的路径
    """
    processor = EquationProcessor(dimensions, vars_list)
    
    # 不再创建备份
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # backup_path = f"{net_file_path}.{timestamp}.bak"
    # shutil.copy2(net_file_path, backup_path)
    
    # 生成代码
    temp_file = os.path.join(os.path.dirname(__file__), "temp_pytorch_code.txt")
    processor.generate_code_for_pytorch_net(list(equations), temp_file)
    
    try:
        with open(temp_file, "r") as f:
            generated_code = f.read()
        
        # 更新文件
        start_marker = "# auto code begin"
        end_marker = "# auto code end"
        
        with open(net_file_path, "r") as f:
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
            raise ValueError(f"在文件 {net_file_path} 中未找到需要替换的代码段")

        with open(net_file_path, "w") as f:
            f.writelines(new_lines)
            
        os.remove(temp_file)
        print(f"已成功更新 {net_file_path}")
        
    except Exception as e:
        # 不再恢复备份，因为没有创建备份
        print(f"更新文件时出错: {str(e)}")
        print("注意：未创建备份文件，原文件可能已被修改")
        raise


def update_hybrid_fitter_code(
    equations: Tuple[str, ...],
    vars_list: List[str],
    dimensions: List[str],
    hybrid_fitter_path: Optional[str] = None,
) -> None:
    """
    更新HybridFitter类中的代码的便捷函数（保持向后兼容性）
    
    Args:
        equations: 方程组
        vars_list: 变量列表
        dimensions: 维度列表
        hybrid_fitter_path: hybrid_fitter.py的路径，默认为None则使用预设路径
    """
    print("警告: update_hybrid_fitter_code 已弃用，请使用 update_physics_loss_from_config")
    # 为了兼容性，保留一个空实现
    pass


# 测试代码
if __name__ == "__main__":
    # 示例用法
    config_path = "cases/linear_pde_cases/poisson_2d_sinpixsinpiy/config.json"
    
    try:
        update_physics_loss_from_config(config_path)
        print("代码生成成功!")
    except Exception as e:
        print(f"代码生成失败: {str(e)}")
