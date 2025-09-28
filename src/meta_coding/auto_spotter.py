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
        InitializeEquationProcessor
        
        Args:
            dimensions: DimensionsList (For example: ['x'] 表示1D, ['x', 'y'] 表示2D)
            vars_list: Variable list (For example: ['u', 'v', 'p'])
        """
        self.dimensions = dimensions
        self.vars_list = vars_list

        # Create符号variable
        self.vars = {dim: sp.Symbol(dim) for dim in dimensions}

        # Create基本variable
        self.var_symbols = {var: sp.Symbol(var) for var in vars_list}

        # CreateDerivatives符号
        self.derivatives = {}
        for var in vars_list:
            for dim in dimensions:
                self.derivatives[f"{var}_{dim}"] = sp.Symbol(f"{var}_{dim}")

    def parse_equation(self, eq_str: str) -> sp.Expr:
        """
        AnalyticalEquation字符串为sympyExpression
        
        Args:
            eq_str: Equation字符串
            
        Returns:
            AnalyticalBackward的sympyExpression
            
        Raises:
            ValueError: EquationAnalyticalError
        """
        if eq_str == "0":
            return sp.Integer(0)

        # 替换DerivativesExpression
        for var in self.vars_list:
            for dim in self.dimensions:
                eq_str = eq_str.replace(f"diff({var},{dim})", f"{var}_{dim}")
                # Process高阶Derivatives
                for order in range(2, 6):  # SupportTo5阶Derivatives
                    old_pattern = f"diff({var},{dim},{order})"
                    new_pattern = f"{var}_" + dim * order
                    eq_str = eq_str.replace(old_pattern, new_pattern)

        # 将Equation字符串Convert为Expression
        if '=' in eq_str:
            lhs, rhs = eq_str.split('=')
            eq_str = f"({lhs}) - ({rhs})"

        try:
            expr = sp.sympify(eq_str)
            return expr
        except Exception as e:
            raise ValueError(f"EquationAnalyticalError: {str(e)}, 原Equation: {eq_str}")

    def _find_used_derivatives(self, equations: Union[List[str], Dict[str, List[str]]]) -> Dict[str, int]:
        """
        AnalyzeEquation中Using的Derivatives及其最高阶数
        
        Args:
            equations: EquationGroup
            
        Returns:
            Using的Derivatives及其最高阶数的Dictionary
        """
        used_derivatives = {}
        
        # UnifyProcessformat
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
                
            # Analyze各种Derivativespattern
            for var in self.vars_list:
                # Checkvariable本身
                if var in eq:
                    key = f"{var}"
                    used_derivatives[key] = max(used_derivatives.get(key, 0), 0)
                
                for dim in self.dimensions:
                    # Check各阶Derivatives
                    for order in range(1, 6):
                        if order == 1:
                            pattern = f"diff({var},{dim})"
                        else:
                            pattern = f"diff({var},{dim},{order})"
                        
                        if pattern in eq:
                            # For高阶Derivatives，也NeedAll低阶Derivatives
                            for lower_order in range(1, order + 1):
                                if lower_order == 1:
                                    key = f"{var}_{dim}"
                                else:
                                    key = f"{var}_{dim}" + dim * (lower_order - 1)
                                used_derivatives[key] = max(used_derivatives.get(key, 0), lower_order)
        
        return used_derivatives

    def generate_pytorch_derivatives(self, equations: Union[List[str], Dict[str, List[str]]]) -> str:
        """
        GeneratePyTorchNeural network的DerivativesCompute代yard
        
        Args:
            equations: EquationGroup
            
        Returns:
            Generate的DerivativesCompute代yard
        """
        used_derivatives = self._find_used_derivatives(equations)
        derivatives_code = []
        
        # Generatevariable提取代yard
        derivatives_code.append("        # Extract physical quantities from output")
        for i, var in enumerate(self.vars_list):
            if any(key.startswith(var) for key in used_derivatives.keys()):
                derivatives_code.append(f"        {var} = U[..., {i}]")
        
        # Generate一阶Derivatives代yard
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
        
        # Generate高阶Derivatives代yard
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
        将Equation字符串Convert为PyTorchformat
        
        Args:
            eq: Equation字符串
            
        Returns:
            PyTorchformat的Equation字符串
        """
        eq_pytorch = eq
        for var in self.vars_list:
            for dim in self.dimensions:
                # Process高阶Derivatives
                for order in range(5, 0, -1):  # From高阶To低阶
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
        GenerateUnify的Operators代yard，适用于AllProblemType

        Args:
            config_dict: ConfigurationDictionary

        Returns:
            Generate的Operators代yard
        """
        operators_code = []

        # Geteqfield
        eq_dict = config_dict.get("eq", {})
        problem_type = config_dict.get("problem_type", "linear_pde")

        # For linear_pde, only process L1 operators, skip S (handled separately in data processing)
        if problem_type == "linear_pde":
            # Only generate L1 operators for linear PDEs
            if "L1" in eq_dict:
                operators_code.append("        # L1 operators")
                eq_list = eq_dict["L1"]

                if isinstance(eq_list, list):
                    eq_terms = []
                    for eq in eq_list:
                        eq_pytorch = self._convert_equation_to_pytorch(eq)
                        eq_terms.append(eq_pytorch)

                    if len(eq_terms) == 1:
                        operators_code.append(f"        L1 = [{eq_terms[0]}]")
                    else:
                        operators_code.append("        L1 = [")
                        for i, term in enumerate(eq_terms):
                            if i == len(eq_terms) - 1:
                                operators_code.append(f"            {term}")
                            else:
                                operators_code.append(f"            {term},")
                        operators_code.append("        ]")
                else:
                    eq_pytorch = self._convert_equation_to_pytorch(eq_list)
                    operators_code.append(f"        L1 = [{eq_pytorch}]")

            # Add empty placeholders for other operators to maintain code structure
            operators_code.append("")
            operators_code.append("        # L2 operators (not used in linear PDEs)")
            operators_code.append("        L2 = []")
            operators_code.append("")
            operators_code.append("        # F operators (not used in linear PDEs)")
            operators_code.append("        F = []")
            operators_code.append("")
            operators_code.append("        # N operators (not used in linear PDEs)")
            operators_code.append("        N = []")

        else:
            # For other problem types, process all operators as before
            for op_name, eq_list in eq_dict.items():
                # Skip S field for all problem types (handled in data processing)
                if op_name == "S":
                    continue

                operators_code.append(f"        # {op_name} operators")

                if isinstance(eq_list, list):
                    # GenerateListformat
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
                    # If不YesList，When作SingleEquationProcess
                    eq_pytorch = self._convert_equation_to_pytorch(eq_list)
                    operators_code.append(f"        {op_name} = [{eq_pytorch}]")

                if op_name != list(eq_dict.keys())[-1]:  # 不YesFinally一个Operators，添加空行
                    operators_code.append("")
        
        # ForTimePDE，还NeedProcessExtra的Operatorsfield
        if config_dict.get("problem_type") == "time_pde":
            # Processing timePDE特有的Operators
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
                    
                    # GenerateListformat
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
        GeneratePyTorchNeural network的Intact代yard，只IncludeDerivatives和OperatorsPartial
        
        Args:
            equations: EquationGroup（NowMain用于DerivativesAnalyze）
            output_path: OutputFilePath
            problem_type: ProblemType
            config_dict: ConfigurationDictionary
            
        Raises:
            Exception: 代yardGenerateError
        """
        try:
            # CollectAllEquation用于DerivativesAnalyze
            all_equations = []
            
            # FromeqfieldCollectEquation
            if config_dict and "eq" in config_dict:
                eq_dict = config_dict["eq"]
                for op_name, eq_list in eq_dict.items():
                    if isinstance(eq_list, list):
                        all_equations.extend(eq_list)
                    else:
                        all_equations.append(eq_list)
            
            # ForTimePDE，还要From其他fieldCollectEquation
            if problem_type == "time_pde" and config_dict:
                time_operators = ["eq_L1", "eq_L2", "f_L2", "N"]
                for op_type in time_operators:
                    if op_type in config_dict and config_dict[op_type]:
                        all_equations.extend(config_dict[op_type])
            
            # GenerateDerivativesCompute代yard
            derivatives_code = self.generate_pytorch_derivatives(all_equations)
            
            # GenerateUnify的Operators代yard
            operators_code = self.generate_unified_operators(config_dict) if config_dict else ""

            # GenerateConfiguration签名（用于Backward续ConsistencyCheck）
            config_signature = ""
            if config_dict:
                # 提取OperatorsConfiguration
                sig_dict = {}
                if "eq" in config_dict:  # 新format
                    eq = config_dict["eq"]
                    sig_dict = {
                        "L1": eq.get("L1", []),
                        "L2": eq.get("L2", []),
                        "F": eq.get("F", []),
                        "N": eq.get("N", [])
                    }
                else:  # 兼容旧format
                    sig_dict = {
                        "L1": config_dict.get("eq_L1", []),
                        "L2": config_dict.get("eq_L2", []),
                        "F": config_dict.get("f_L2", []),
                        "N": config_dict.get("N", [])
                    }

                import json
                sig_str = json.dumps(sig_dict, sort_keys=True)
                config_signature = f"# Config signature: {sig_str}\n"

            # Combination代yard
            code = f"""# auto code begin
{config_signature}{derivatives_code}

{operators_code}

# auto code end"""
            
            # WriteFile
            with open(output_path, "w") as f:
                f.write(code)

            print(f"PyTorchNeural network代yard已GenerateTo: {output_path}")

        except Exception as e:
            print(f"GeneratePyTorch代yard时Exit错: {str(e)}")
            raise


class AutoCodeGenerator:
    """自动代yardGeneratorclass"""
    def __init__(self, config_path: str):
        """
        Initialize自动代yardGenerator
        
        Args:
            config_path: ConfigurationFilePath
        """
        self.config_path = config_path
        self.config_dict = self._load_config()
        self.problem_type = self.config_dict.get("problem_type", "linear_pde")
        self.dimensions = self.config_dict.get("spatial_vars", ["x", "y"])
        self.vars_list = self.config_dict.get("vars_list", ["u"])
        
        self.processor = EquationProcessor(self.dimensions, self.vars_list)

    def _load_config(self) -> Dict:
        """LoadingConfigurationFile"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"LoadingConfigurationFileFail: {str(e)}")

    def check_config_net_consistency(self, net_file_path: str) -> Tuple[bool, str]:
        """
        CheckConfigurationFile中的OperatorsDefinition与net.py中Generate代yard的Consistency

        Args:
            net_file_path: net.pyFilePath

        Returns:
            (NeedRe-Generate, 原因description)
        """
        # Checknet.pyYesNo存At
        if not os.path.exists(net_file_path):
            return True, "net.pyFile不存At"

        # Readnet.pycontent
        with open(net_file_path, 'r') as f:
            net_content = f.read()

        # CheckYesNo有auto codeBlock
        if "# auto code begin" not in net_content or "# auto code end" not in net_content:
            return True, "net.py中没有auto codeMark"

        # 提取auto codeBlock
        begin_idx = net_content.find("# auto code begin")
        end_idx = net_content.find("# auto code end")
        auto_code_block = net_content[begin_idx:end_idx]

        # Checkauto codeBlockYesNo为空
        lines = auto_code_block.split('\n')[1:-1]
        has_code = any(line.strip() and not line.strip().startswith('#') for line in lines)

        if not has_code:
            return True, "auto codeBlock为空，NeedGenerate代yard"

        # 提取CurrentConfiguration的Operators签名
        config_signature = self._extract_config_signature()

        # Atauto codeBlock中SearchConfiguration签名
        if "# Config signature:" in auto_code_block:
            # 提取Save的签名
            for line in lines:
                if "# Config signature:" in line:
                    saved_sig = line.split("# Config signature:")[1].strip()
                    try:
                        saved_dict = json.loads(saved_sig)
                        current_dict = json.loads(config_signature)

                        # Compare两个签名
                        if saved_dict == current_dict:
                            return False, "config与net.py代yard一致"
                        else:
                            # 找ExitSpecific的差异
                            diff_msg = self._find_signature_diff(saved_dict, current_dict)
                            return True, f"configOperatorsDefinition已更改: {diff_msg}"
                    except:
                        return True, "无法AnalyticalSave的签名"

        # If没有签名，Pass更Abstract的方式Check
        return self._abstract_consistency_check(auto_code_block, config_signature)

    def _extract_config_signature(self) -> str:
        """提取ConfigurationFile的Operators签名"""
        sig_dict = {}

        if "eq" in self.config_dict:  # 新format
            eq = self.config_dict["eq"]
            sig_dict = {
                "L1": eq.get("L1", []),
                "L2": eq.get("L2", []),
                "F": eq.get("F", []),
                "N": eq.get("N", [])
            }
        else:  # 兼容旧format
            sig_dict = {
                "L1": self.config_dict.get("eq_L1", []),
                "L2": self.config_dict.get("eq_L2", []),
                "F": self.config_dict.get("f_L2", []),
                "N": self.config_dict.get("N", [])
            }

        return json.dumps(sig_dict, sort_keys=True)

    def _find_signature_diff(self, saved_dict: Dict, current_dict: Dict) -> str:
        """找Exit两个签名Dictionary的差异"""
        diffs = []

        for key in ["L1", "L2", "F", "N"]:
            saved = saved_dict.get(key, [])
            current = current_dict.get(key, [])

            if saved != current:
                if not saved and current:
                    diffs.append(f"{key}添加了Operators")
                elif saved and not current:
                    diffs.append(f"{key}删ExceptOperators")
                else:
                    diffs.append(f"{key}Operators已Modification")

        return ", ".join(diffs) if diffs else "未知差异"

    def _abstract_consistency_check(self, auto_code_block: str, config_signature: str) -> Tuple[bool, str]:
        """
        Abstract的ConsistencyCheck，不Dependency签名
        PassAnalyze代yardstructure和OperatorspatternEnter行判断
        """
        try:
            config_dict = json.loads(config_signature)

            # Check每种OperatorsType
            for op_type, ops in config_dict.items():
                if not ops:
                    continue

                # According toOperatorsType确定代yard中的Mark
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

                # At代yard中Search对应的OperatorsDefinition
                match = re.search(pattern, auto_code_block, re.DOTALL)

                if not match and ops:
                    return True, f"{op_type}OperatorsAt代yard中未Find"

                if match:
                    code_content = match.group(1).strip()

                    # 简单Check：Operators数量YesNo一致
                    # PassCompute逗号数量ComeEstimationOperators个数
                    expected_count = len(ops)
                    if code_content:
                        # 简单Estimation：If有content，At least有一个Operators
                        code_has_content = bool(code_content and not code_content.isspace())
                        if expected_count > 0 and not code_has_content:
                            return True, f"{op_type}OperatorsDefinition不匹配"
                    elif expected_count > 0:
                        return True, f"{op_type}OperatorsAt代yard中为空"

            # Default认为一致
            return False, "PassAbstractCheck，代yard与Configuration基本一致"

        except Exception as e:
            # Exit错时保守Process，认为NeedRe-Generate
            return True, f"ConsistencyCheckExit错: {str(e)}"

    def _get_net_file_path(self) -> str:
        """According toProblemType确定net.pyFilePath"""
        case_dir = os.path.dirname(self.config_path)
        
        if self.problem_type == "linear_pde":
            return "src/problem_solvers/linear_pde_solver/core/net.py"
        elif self.problem_type == "time_pde":
            return "src/problem_solvers/time_pde_solver/core/net.py"
        elif self.problem_type == "func_fitting":
            return "src/problem_solvers/func_fitting_solver/core/net.py"
        else:
            raise ValueError(f"不Support的ProblemType: {self.problem_type}")

    def _get_equations(self) -> Union[List[str], Dict[str, List[str]]]:
        """GetEquationGroup用于DerivativesAnalyze"""
        all_equations = []
        
        # FromeqfieldCollectEquation
        if "eq" in self.config_dict and self.config_dict["eq"]:
            eq_dict = self.config_dict["eq"]
            for op_name, eq_list in eq_dict.items():
                if isinstance(eq_list, list):
                    all_equations.extend(eq_list)
                else:
                    all_equations.append(eq_list)
        
        # ForTimePDE，还要From其他fieldCollectEquation
        if self.problem_type == "time_pde":
            time_operators = ["eq_L1", "eq_L2", "f_L2", "N"]
            for op_type in time_operators:
                if op_type in self.config_dict and self.config_dict[op_type]:
                    all_equations.extend(self.config_dict[op_type])
        
        return all_equations

    def update_code(self) -> None:
        """
        UpdateNeural networkFile中的代yard
        
        Raises:
            FileNotFoundError: 找Less than目标File
            ValueError: 未FindNeed替换的代yard段
            Exception: 其他UpdateError
        """
        try:
            net_file_path = self._get_net_file_path()
            
            if not os.path.exists(net_file_path):
                raise FileNotFoundError(f"找Less thanFile: {net_file_path}")

            # 不再Create备份
            # backup_path = self._backup_file(net_file_path)

            # Generate和Update代yard
            self._generate_and_update_code(net_file_path, None)
            
        except Exception as e:
            print(f"Update代yard时Exit错: {str(e)}")
            raise

    def _backup_file(self, file_path: str) -> str:
        """
        CreateFile备份
        
        Args:
            file_path: 要备份的FilePath
            
        Returns:
            备份FilePath
        """
        # 不再Create备份File
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # backup_path = f"{file_path}.{timestamp}.bak"
        # shutil.copy2(file_path, backup_path)
        print(f"跳过备份Create (已Disable)")
        return file_path  # Return原FilePath

    def _generate_and_update_code(self, net_file_path: str, backup_path: str = None) -> None:
        """
        Generate并Update代yard

        Args:
            net_file_path: 目标FilePath
            backup_path: 备份FilePath（已废弃，保留Parameter以兼容）
            
        Raises:
            Exception: 代yardGenerate或UpdateError
        """
        # GenerateTemporary代yardFile
        temp_file = os.path.join(os.path.dirname(__file__), "temp_pytorch_code.txt")
        equations = self._get_equations()
        
        self.processor.generate_code_for_pytorch_net(
            equations, 
            temp_file, 
            self.problem_type,
            self.config_dict
        )

        try:
            # ReadGenerate的代yard
            with open(temp_file, "r") as f:
                generated_code = f.read()

            # Update原File
            self._update_file_content(net_file_path, generated_code)
            
            # CleanupTemporaryFile
            os.remove(temp_file)
            print(f"已SuccessUpdate {net_file_path}")
            
        except Exception as e:
            # 不再Redo备份，Because没有Create备份
            print(f"UpdateFile时Exit错: {str(e)}")
            print("Note：未Create备份File，原File可能已被Modification")
            raise

    def _update_file_content(self, file_path: str, generated_code: str) -> None:
        """
        UpdateFilecontent
        
        Args:
            file_path: FilePath
            generated_code: Generate的代yard
            
        Raises:
            ValueError: 未FindNeed替换的代yard段
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
            raise ValueError(f"AtFile {file_path} 中未FindNeed替换的代yard段")

        with open(file_path, "w") as f:
            f.writelines(new_lines)


def update_physics_loss_from_config(config_path: str) -> None:
    """
    FromConfigurationFile自动Updatephysics loss代yard的便捷function
    
    Args:
        config_path: ConfigurationFilePath
    """
    generator = AutoCodeGenerator(config_path)
    generator.update_code()


# 保留原有function以MaintainTowardBackwardCompatibility
def update_pytorch_net_code(
    equations: Tuple[str, ...],
    vars_list: List[str],
    dimensions: List[str],
    net_file_path: str,
) -> None:
    """
    UpdatePyTorchNeural networkclass中的代yard的便捷function（MaintainTowardBackwardCompatibility）
    
    Args:
        equations: EquationGroup
        vars_list: Variable list
        dimensions: DimensionsList
        net_file_path: net.py的Path
    """
    processor = EquationProcessor(dimensions, vars_list)
    
    # 不再Create备份
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # backup_path = f"{net_file_path}.{timestamp}.bak"
    # shutil.copy2(net_file_path, backup_path)
    
    # Generate代yard
    temp_file = os.path.join(os.path.dirname(__file__), "temp_pytorch_code.txt")
    processor.generate_code_for_pytorch_net(list(equations), temp_file)
    
    try:
        with open(temp_file, "r") as f:
            generated_code = f.read()
        
        # UpdateFile
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
            raise ValueError(f"AtFile {net_file_path} 中未FindNeed替换的代yard段")

        with open(net_file_path, "w") as f:
            f.writelines(new_lines)
            
        os.remove(temp_file)
        print(f"已SuccessUpdate {net_file_path}")
        
    except Exception as e:
        # 不再Redo备份，Because没有Create备份
        print(f"UpdateFile时Exit错: {str(e)}")
        print("Note：未Create备份File，原File可能已被Modification")
        raise


def update_hybrid_fitter_code(
    equations: Tuple[str, ...],
    vars_list: List[str],
    dimensions: List[str],
    hybrid_fitter_path: Optional[str] = None,
) -> None:
    """
    UpdateHybridFitterclass中的代yard的便捷function（MaintainTowardBackwardCompatibility）
    
    Args:
        equations: EquationGroup
        vars_list: Variable list
        dimensions: DimensionsList
        hybrid_fitter_path: hybrid_fitter.py的Path，Default为None则Using预设Path
    """
    print("Warning: update_hybrid_fitter_code 已弃用，请Using update_physics_loss_from_config")
    # 为了Compatibility，保留一个空Implementation
    pass


# Test代yard
if __name__ == "__main__":
    # exampleUsage
    config_path = "cases/linear_pde_cases/poisson_2d_sinpixsinpiy/config.json"
    
    try:
        update_physics_loss_from_config(config_path)
        print("代yardGenerateSuccess!")
    except Exception as e:
        print(f"代yardGenerateFail: {str(e)}")
