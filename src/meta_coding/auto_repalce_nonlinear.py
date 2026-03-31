import os
from typing import List, Optional, Tuple
import sympy as sp

from .nonlinear_codegen_update_mixin import NonlinearCodegenUpdateMixin
from .nonlinear_equation_codegen_mixin import NonlinearEquationCodegenMixin


class EquationProcessor(NonlinearEquationCodegenMixin):
    def __init__(self, dimensions: List[str], vars_list: List[str]):
        self.dimensions = dimensions
        self.vars_list = vars_list
        self.vars = {dim: sp.Symbol(dim) for dim in dimensions}
        self.var_symbols = {var: sp.Symbol(var) for var in vars_list}
        self.derivatives = {}
        for var in vars_list:
            for dim in dimensions:
                self.derivatives[f"{var}_{dim}"] = sp.Symbol(f"{var}_{dim}")

class NonlinearCodeGenerator(NonlinearCodegenUpdateMixin):
    def __init__(self, dimensions: List[str], vars_list: List[str], hybrid_fitter_path: Optional[str] = None):
        self.dimensions = dimensions
        self.vars_list = vars_list
        self.processor = EquationProcessor(dimensions, vars_list)
        self.hybrid_fitter_path = hybrid_fitter_path or "NewFramework/NS_stable/models/hybrid_fitter.py"

    def update_code(self, equations: Tuple[str, ...]) -> None:
        try:
            if not os.path.exists(self.hybrid_fitter_path):
                raise FileNotFoundError(f"找不到文件: {self.hybrid_fitter_path}")

            backup_path = self._backup_file()
            self._generate_and_update_code(equations, backup_path)
        except Exception as exc:
            print(f"更新代码时出错: {str(exc)}")
            raise


def update_hybrid_fitter_code(
    equations: Tuple[str, ...],
    vars_list: List[str],
    dimensions: List[str],
    hybrid_fitter_path: Optional[str] = None,
) -> None:
    generator = NonlinearCodeGenerator(dimensions, vars_list, hybrid_fitter_path)
    generator.update_code(equations)
