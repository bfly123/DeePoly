from typing import Optional

import numpy as np

from src.abstract_class.config.data_helpers import (
    MathExpressionEvaluator,
    ReferenceSolutionLoader,
    SourceTermResolver,
)


class BaseDataResolutionMixin:
    def _parse_math_expression(self, expr: str, x: np.ndarray) -> np.ndarray:
        return self._expression_evaluator.evaluate(expr, x)

    def _parse_piecewise_expression(self, value_dict: dict, x: np.ndarray) -> np.ndarray:
        return self._expression_evaluator.evaluate_piecewise(value_dict, x)

    def _load_source_term(self, x_global: np.ndarray) -> np.ndarray:
        return self._build_source_term_resolver().resolve(
            self.config,
            self.custom_data_generator,
            x_global,
            self.n_eqs,
        )

    def _is_file_reference(self, source_expr: str) -> bool:
        return self._build_source_term_resolver().is_file_reference(source_expr)

    def _load_source_from_file(self, file_name: str, x_global: np.ndarray) -> np.ndarray:
        return self._build_source_term_resolver().load_source_from_file(
            file_name,
            self.custom_data_generator,
            x_global,
        )

    def load_reference_solution(self, x_global: np.ndarray) -> Optional[np.ndarray]:
        return self._build_reference_solution_loader().load(
            self.config,
            self.custom_data_generator,
            x_global,
            self.n_eqs,
        )

    def _load_reference_from_mat_file(
        self,
        file_path: str,
        x_global: np.ndarray,
    ) -> Optional[np.ndarray]:
        del x_global
        return self._build_reference_solution_loader().load_from_mat_file(
            file_path,
            getattr(self.config, "case_dir", None),
        )

    def _load_reference_from_file(
        self,
        file_name: str,
        x_global: np.ndarray,
    ) -> Optional[np.ndarray]:
        return self._build_reference_solution_loader().load_from_file(
            file_name,
            self.custom_data_generator,
            x_global,
        )

    def _build_source_term_resolver(self) -> SourceTermResolver:
        return SourceTermResolver(self._expression_evaluator)

    def _build_reference_solution_loader(self) -> ReferenceSolutionLoader:
        return ReferenceSolutionLoader(self._expression_evaluator)

    def _build_expression_evaluator(self) -> MathExpressionEvaluator:
        return MathExpressionEvaluator(self.config.spatial_vars)
