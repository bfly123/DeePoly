import os
from typing import Optional

import numpy as np
import scipy.io

from src.abstract_class.config.data_expression_evaluator import (
    MathExpressionEvaluator,
    normalize_column_vector,
)


class SourceTermResolver:
    """Resolve source terms from config expressions or case-local modules."""

    def __init__(self, expression_evaluator: MathExpressionEvaluator):
        self.expression_evaluator = expression_evaluator

    def resolve(self, config, custom_data_generator, x_global: np.ndarray, n_eqs: int) -> np.ndarray:
        n_points = x_global.shape[0]
        if hasattr(config, "eq") and isinstance(config.eq, dict) and "S" in config.eq:
            return self._resolve_eq_source(config.eq["S"], custom_data_generator, x_global, n_eqs, n_points)
        if hasattr(config, "source_term"):
            return self._resolve_source_term(config.source_term, custom_data_generator, x_global, n_eqs)
        return np.zeros((n_points, n_eqs))

    def _resolve_eq_source(
        self,
        s_field,
        custom_data_generator,
        x_global: np.ndarray,
        n_eqs: int,
        n_points: int,
    ) -> np.ndarray:
        if isinstance(s_field, str):
            return self._resolve_string_source(s_field, custom_data_generator, x_global, n_eqs)
        if isinstance(s_field, list) and s_field:
            return self._resolve_source_list(s_field, x_global, n_eqs, n_points)
        return np.zeros((n_points, n_eqs))

    def _resolve_string_source(
        self,
        source_expr: str,
        custom_data_generator,
        x_global: np.ndarray,
        n_eqs: int,
    ) -> np.ndarray:
        if self.is_file_reference(source_expr):
            source_single = self.load_source_from_file(source_expr, custom_data_generator, x_global)
        else:
            source_single = self.expression_evaluator.evaluate(source_expr, x_global)
        return np.tile(source_single, (1, n_eqs))

    def _resolve_source_list(
        self,
        source_fields,
        x_global: np.ndarray,
        n_eqs: int,
        n_points: int,
    ) -> np.ndarray:
        source_results = []
        for index in range(n_eqs):
            source_expr = source_fields[index]
            if source_expr == "0":
                source_results.append(np.zeros((n_points, 1)))
                continue
            source_results.append(self.expression_evaluator.evaluate(source_expr, x_global))
        return np.hstack(source_results)

    def _resolve_source_term(
        self,
        source_term,
        custom_data_generator,
        x_global: np.ndarray,
        n_eqs: int,
    ) -> np.ndarray:
        if isinstance(source_term, str):
            source_single = self.expression_evaluator.evaluate(source_term, x_global)
            return np.tile(source_single, (1, n_eqs))
        if source_term is True:
            if hasattr(custom_data_generator, "generate_source_term"):
                return custom_data_generator.generate_source_term(x_global)
            raise ValueError("Custom source term generator not found")
        return np.zeros((x_global.shape[0], n_eqs))

    def is_file_reference(self, source_expr: str) -> bool:
        return source_expr.endswith(".py")

    def load_source_from_file(
        self,
        file_name: str,
        custom_data_generator,
        x_global: np.ndarray,
    ) -> np.ndarray:
        if not custom_data_generator:
            raise ValueError(f"Cannot load source from {file_name}: custom data generator not available")
        if not hasattr(custom_data_generator, "generate_source_term"):
            raise ValueError(f"Function 'generate_source_term' not found in {file_name}")
        return normalize_column_vector(custom_data_generator.generate_source_term(x_global))


class ReferenceSolutionLoader:
    """Load reference solutions from expressions, .mat files, or case-local modules."""

    def __init__(self, expression_evaluator: MathExpressionEvaluator):
        self.expression_evaluator = expression_evaluator

    def load(self, config, custom_data_generator, x_global: np.ndarray, n_eqs: int) -> Optional[np.ndarray]:
        if not hasattr(config, "reference_solution") or not config.reference_solution:
            return None
        ref_field = config.reference_solution
        if isinstance(ref_field, str):
            return self._load_string_reference(ref_field, config, custom_data_generator, x_global, n_eqs)
        return None

    def _load_string_reference(
        self,
        ref_field: str,
        config,
        custom_data_generator,
        x_global: np.ndarray,
        n_eqs: int,
    ) -> Optional[np.ndarray]:
        if ref_field.endswith((".mat", ".m")):
            return self.load_from_mat_file(ref_field, getattr(config, "case_dir", None))
        if ref_field.endswith(".py"):
            return self.load_from_file(ref_field, custom_data_generator, x_global)
        ref_single = self.expression_evaluator.evaluate(ref_field, x_global)
        return np.tile(ref_single, (1, n_eqs))

    def load_from_mat_file(self, file_path: str, case_dir: Optional[str]) -> Optional[np.ndarray]:
        resolved_path = self._resolve_mat_path(file_path, case_dir)
        if resolved_path is None:
            return None
        try:
            data = scipy.io.loadmat(resolved_path)
            return self._extract_mat_solution(data)
        except Exception:
            return None

    def _resolve_mat_path(self, file_path: str, case_dir: Optional[str]) -> Optional[str]:
        if not os.path.isabs(file_path):
            file_path = os.path.join(case_dir or os.getcwd(), file_path)
        if file_path.endswith(".m"):
            file_path = file_path.replace(".m", ".mat")
        if not os.path.exists(file_path):
            return None
        return file_path

    def _extract_mat_solution(self, data: dict) -> Optional[np.ndarray]:
        if "usol" in data:
            return normalize_column_vector(data["usol"][:, -1:])
        if "u" in data:
            return normalize_column_vector(data["u"])
        return None

    def load_from_file(self, file_name: str, custom_data_generator, x_global: np.ndarray) -> Optional[np.ndarray]:
        if not custom_data_generator:
            return None
        if not hasattr(custom_data_generator, "generate_reference_solution"):
            return None
        try:
            values = custom_data_generator.generate_reference_solution(x_global)
            return normalize_column_vector(values)
        except Exception:
            return None
