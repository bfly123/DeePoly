import json
import os
from typing import Dict, List, Optional, Tuple

from src.abstract_class.config.config_io import load_config_file

from .spotter_codegen import (
    abstract_consistency_check,
    build_config_signature,
    collect_equations,
    compose_pytorch_code,
    extract_auto_code_block,
    extract_saved_signature,
    find_signature_diff,
    has_generated_code,
    replace_auto_code_section,
)
from .spotter_equation_processor import EquationProcessor


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
            return load_config_file(self.config_path)
        except Exception as exc:
            raise ValueError(f"LoadingConfigurationFileFail: {exc}") from exc

    def check_config_net_consistency(self, net_file_path: str) -> Tuple[bool, str]:
        """
        CheckConfigurationFile中的OperatorsDefinition与net.py中Generate代yard的Consistency
        """
        if not os.path.exists(net_file_path):
            return True, "net.pyFile不存At"

        with open(net_file_path, "r") as net_file:
            net_content = net_file.read()

        auto_code_block = extract_auto_code_block(net_content)
        if auto_code_block is None:
            return True, "net.py中没有auto codeMark"

        if not has_generated_code(auto_code_block):
            return True, "auto codeBlock为空，NeedGenerate代yard"

        config_signature = self._extract_config_signature()
        saved_signature = extract_saved_signature(auto_code_block)
        if saved_signature is None:
            return self._abstract_consistency_check(auto_code_block, config_signature)

        try:
            current_signature = json.loads(config_signature)
        except Exception:
            return True, "无法AnalyticalSave的签名"

        if saved_signature == current_signature:
            return False, "config与net.py代yard一致"

        diff_msg = self._find_signature_diff(saved_signature, current_signature)
        return True, f"configOperatorsDefinition已更改: {diff_msg}"

    def _extract_config_signature(self) -> str:
        """提取ConfigurationFile的Operators签名"""
        return build_config_signature(self.config_dict, self.problem_type)

    def _find_signature_diff(self, saved_dict: Dict, current_dict: Dict) -> str:
        """找Exit两个签名Dictionary的差异"""
        return find_signature_diff(saved_dict, current_dict)

    def _abstract_consistency_check(
        self,
        auto_code_block: str,
        config_signature: str,
    ) -> Tuple[bool, str]:
        """
        Abstract的ConsistencyCheck，不Dependency签名
        """
        return abstract_consistency_check(auto_code_block, config_signature)

    def _get_net_file_path(self) -> str:
        """Unified generation to base_net.py - no problem_type detection needed"""
        return "src/abstract_class/base_net.py"

    def _get_equations(self) -> List[str]:
        """GetEquationGroup用于DerivativesAnalyze"""
        return collect_equations(self.config_dict, self.problem_type)

    def update_code(self) -> None:
        """
        UpdateNeural networkFile中的代yard
        """
        try:
            net_file_path = self._get_net_file_path()
            if not os.path.exists(net_file_path):
                raise FileNotFoundError(f"找Less thanFile: {net_file_path}")
            self._generate_and_update_code(net_file_path, None)
        except Exception as exc:
            print(f"Update代yard时Exit错: {exc}")
            raise

    def _backup_file(self, file_path: str) -> str:
        """CreateFile备份"""
        print("跳过备份Create (已Disable)")
        return file_path

    def _generate_and_update_code(self, net_file_path: str, backup_path: str = None) -> None:
        """
        Generate并Update代yard
        """
        _ = backup_path
        equations = self._get_equations()
        generated_code = compose_pytorch_code(
            processor=self.processor,
            equations=equations,
            problem_type=self.problem_type,
            config_dict=self.config_dict,
        )

        try:
            self._update_file_content(net_file_path, generated_code)
            print(f"已SuccessUpdate {net_file_path}")
        except Exception as exc:
            print(f"UpdateFile时Exit错: {exc}")
            print("Note：未Create备份File，原File可能已被Modification")
            raise

    def _update_file_content(self, file_path: str, generated_code: str) -> None:
        """
        UpdateFilecontent
        """
        replace_auto_code_section(file_path, generated_code)


def update_physics_loss_from_config(config_path: str) -> None:
    """
    FromConfigurationFile自动Updatephysics loss代yard的便捷function
    """
    generator = AutoCodeGenerator(config_path)
    generator.update_code()


def update_pytorch_net_code(
    equations: Tuple[str, ...],
    vars_list: List[str],
    dimensions: List[str],
    net_file_path: str,
) -> None:
    """
    UpdatePyTorchNeural networkclass中的代yard的便捷function（MaintainTowardBackwardCompatibility）
    """
    processor = EquationProcessor(dimensions, vars_list)
    generated_code = compose_pytorch_code(processor=processor, equations=list(equations))

    try:
        replace_auto_code_section(net_file_path, generated_code)
        print(f"已SuccessUpdate {net_file_path}")
    except Exception as exc:
        print(f"UpdateFile时Exit错: {exc}")
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
    """
    _ = (equations, vars_list, dimensions, hybrid_fitter_path)
    print("Warning: update_hybrid_fitter_code 已弃用，请Using update_physics_loss_from_config")
