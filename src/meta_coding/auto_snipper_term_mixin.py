import re
from typing import Any, Dict, List


class OperatorParserTermMixin:
    def replace_with_names(self, expr: str) -> str:
        processed = re.sub(r"diff\((\w+),(\w+)(?:,(\d+))?\)", self._replace_diff_match, expr)
        for var in self.variables:
            processed = re.sub(rf"\b{var}\b", var.upper(), processed)
        return processed

    def _replace_diff_match(self, match) -> str:
        var_name = match.group(1)
        spatial_var = match.group(2)
        order = int(match.group(3)) if match.group(3) else 1
        name = var_name.upper()
        if order > 0:
            name += "_" + spatial_var * order
        return name

    def decompose_terms(
        self,
        expr: str,
        derivatives: Dict[str, List[int]],
    ) -> Dict[str, Any]:
        if not expr:
            return {"derivative_indices": [], "symbolic_expr": expr}
        processed = self.replace_with_names(expr)
        indices = {
            deriv_idx
            for name, (_, deriv_idx) in derivatives.items()
            if re.search(rf"\b{name}\b", processed)
        }
        return {"derivative_indices": sorted(indices), "symbolic_expr": processed}

    def parse(self, operators: Dict[str, List[str]]) -> Dict[str, Any]:
        all_expressions = [expr for equations in operators.values() for expr in equations]
        derivatives = self.extract_unique_derivatives(all_expressions)
        all_derivatives = self.extract_all_derivatives_new(all_expressions, derivatives)
        max_orders = self.get_max_orders(all_expressions)
        operator_terms = {
            op_name: [self.decompose_terms(expr, all_derivatives) for expr in equations]
            for op_name, equations in operators.items()
        }
        return {
            "derivatives": derivatives,
            "all_derivatives": all_derivatives,
            "max_derivative_orders": max_orders,
            "operator_terms": operator_terms,
        }
