import re
from typing import Dict, List


class OperatorParserDerivativeMixin:
    def extract_unique_derivatives(self, expressions: List[str]) -> List[List[int]]:
        unique_patterns = {tuple([0] * len(self.spatial_vars))}
        for expr in expressions:
            for pattern in self._expression_derivative_patterns(expr):
                unique_patterns.add(pattern)
        return [list(pattern) for pattern in sorted(unique_patterns)]

    def _expression_derivative_patterns(self, expr: str):
        for var_name, spatial_var, order_str in re.findall(r"diff\((\w+),(\w+)(?:,(\d+))?\)", expr):
            if var_name not in self.variables or spatial_var not in self.spatial_vars:
                continue
            spatial_idx = self.spatial_vars.index(spatial_var)
            order = int(order_str) if order_str else 1
            pattern = [0] * len(self.spatial_vars)
            pattern[spatial_idx] = order
            yield tuple(pattern)

    def extract_all_derivatives_new(
        self,
        expressions: List[str],
        derivatives: List[List[int]],
    ) -> Dict[str, List[int]]:
        all_derivatives = {}
        for var_idx, var in enumerate(self.variables):
            for deriv_idx, deriv_pattern in enumerate(derivatives):
                if not self._pattern_used_in_expressions(var, deriv_pattern, expressions):
                    continue
                name = self._build_derivative_name(var, deriv_pattern)
                all_derivatives[name] = [var_idx, deriv_idx]
        return all_derivatives

    def _pattern_used_in_expressions(
        self,
        var: str,
        deriv_pattern: List[int],
        expressions: List[str],
    ) -> bool:
        if self._is_zero_order_pattern(deriv_pattern):
            return any(re.search(rf"\b{var}\b(?!\s*[\,\)])", expr) for expr in expressions)
        regexes = self._pattern_regexes(var, deriv_pattern)
        return any(re.search(regex, expr) for regex in regexes for expr in expressions)

    @staticmethod
    def _is_zero_order_pattern(deriv_pattern: List[int]) -> bool:
        return all(order == 0 for order in deriv_pattern)

    def _pattern_regexes(self, var: str, deriv_pattern: List[int]) -> List[str]:
        regexes = []
        for spatial_idx, order in enumerate(deriv_pattern):
            if order <= 0:
                continue
            spatial_var = self.spatial_vars[spatial_idx]
            if order == 1:
                regexes.append(rf"diff\({var},{spatial_var}\)")
            else:
                regexes.append(rf"diff\({var},{spatial_var},{order}\)")
        return regexes

    def _build_derivative_name(self, var: str, deriv_pattern: List[int]) -> str:
        name = var.upper()
        for spatial_idx, order in enumerate(deriv_pattern):
            if order > 0:
                name += "_" + self.spatial_vars[spatial_idx] * order
        return name

    def extract_all_derivatives(self, expressions: List[str]) -> Dict[str, List[int]]:
        derivatives = self._zero_order_derivatives()
        for expr in expressions:
            self._record_expression_derivatives(derivatives, expr)
            self._record_standalone_variables(derivatives, expr)
        return derivatives

    def _zero_order_derivatives(self) -> Dict[str, List[int]]:
        return {
            var.upper(): [index] + [0] * len(self.spatial_vars)
            for index, var in enumerate(self.variables)
        }

    def _record_expression_derivatives(
        self,
        derivatives: Dict[str, List[int]],
        expr: str,
    ) -> None:
        for var_name, spatial_var, order_str in re.findall(r"diff\((\w+),(\w+)(?:,(\d+))?\)", expr):
            if var_name not in self.variables or spatial_var not in self.spatial_vars:
                continue
            var_idx = self.variables.index(var_name)
            spatial_idx = self.spatial_vars.index(spatial_var)
            order = int(order_str) if order_str else 1
            pattern = [0] * len(self.spatial_vars)
            pattern[spatial_idx] = order
            derivatives[self._build_derivative_name(var_name, pattern)] = [var_idx] + pattern

    def _record_standalone_variables(
        self,
        derivatives: Dict[str, List[int]],
        expr: str,
    ) -> None:
        for var in self.variables:
            if re.search(rf"\b{var}\b(?!\s*[\,\)])", expr):
                derivatives[var.upper()] = [self.variables.index(var)] + [0] * len(self.spatial_vars)

    def get_max_orders(self, expressions: List[str]) -> List[List[int]]:
        max_orders = [[0] * len(self.spatial_vars) for _ in self.variables]
        for expr in expressions:
            self._update_expression_max_orders(max_orders, expr)
        return max_orders

    def _update_expression_max_orders(self, max_orders: List[List[int]], expr: str) -> None:
        for var_name, spatial_var, order_str in re.findall(r"diff\((\w+),(\w+)(?:,(\d+))?\)", expr):
            if var_name not in self.variables or spatial_var not in self.spatial_vars:
                continue
            var_idx = self.variables.index(var_name)
            spatial_idx = self.spatial_vars.index(spatial_var)
            order = int(order_str) if order_str else 1
            max_orders[var_idx][spatial_idx] = max(max_orders[var_idx][spatial_idx], order)
