from typing import Dict, List, Optional, Tuple, Union

import sympy as sp


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
        self.vars = {dim: sp.Symbol(dim) for dim in dimensions}
        self.var_symbols = {var: sp.Symbol(var) for var in vars_list}
        self.derivatives = self._build_derivative_symbols()

    def _build_derivative_symbols(self) -> Dict[str, sp.Symbol]:
        derivatives = {}
        for var in self.vars_list:
            for dim in self.dimensions:
                derivatives[f"{var}_{dim}"] = sp.Symbol(f"{var}_{dim}")
        return derivatives

    def _replace_derivative_tokens(self, eq_str: str, prefix: str = "") -> str:
        converted = eq_str
        for var in self.vars_list:
            for dim in self.dimensions:
                for order in range(5, 0, -1):
                    old_pattern, new_pattern = self._derivative_replacement(var, dim, order, prefix)
                    converted = converted.replace(old_pattern, new_pattern)
        return converted

    def _derivative_replacement(
        self,
        var: str,
        dim: str,
        order: int,
        prefix: str = "",
    ) -> Tuple[str, str]:
        if order == 1:
            return f"diff({var},{dim})", f"{prefix}{var}_{dim}"
        return f"diff({var},{dim},{order})", f"{prefix}{var}_{dim}" + dim * (order - 1)

    def _normalize_equations(self, equations: Union[List[str], Dict[str, List[str]]]) -> List[str]:
        if isinstance(equations, dict):
            eq_list = []
            for eq_items in equations.values():
                if isinstance(eq_items, list):
                    eq_list.extend(eq_items)
                else:
                    eq_list.append(eq_items)
            return eq_list
        return list(equations)

    def _record_used_derivative(
        self,
        used_derivatives: Dict[str, int],
        var: str,
        dim: str,
        order: int,
    ) -> None:
        for lower_order in range(1, order + 1):
            suffix = dim * (lower_order - 1)
            key = f"{var}_{dim}" if lower_order == 1 else f"{var}_{dim}{suffix}"
            used_derivatives[key] = max(used_derivatives.get(key, 0), lower_order)

    def _build_variable_extract_lines(self, used_derivatives: Dict[str, int]) -> List[str]:
        lines = ["        # Extract physical quantities from output"]
        for index, var in enumerate(self.vars_list):
            if any(key.startswith(var) for key in used_derivatives):
                lines.append(f"        {var} = U[..., {index}]")
        return lines

    def _build_first_order_derivative_lines(self, used_derivatives: Dict[str, int]) -> List[str]:
        lines = []
        first_order_added = False
        for var in self.vars_list:
            for dim_index, dim in enumerate(self.dimensions):
                deriv_key = f"{var}_{dim}"
                if deriv_key not in used_derivatives:
                    continue
                if not first_order_added:
                    lines.extend(["", "        # Calculate derivatives in each direction"])
                    first_order_added = True
                lines.append(
                    f"        d{var}_{dim} = self.gradients({var}, x_train)[0][..., {dim_index}]"
                )
        return lines

    def _build_higher_order_derivative_lines(self, used_derivatives: Dict[str, int]) -> List[str]:
        lines = []
        for order in range(2, 6):
            order_added = False
            for var in self.vars_list:
                for dim1 in self.dimensions:
                    for target_dim_idx, dim2 in enumerate(self.dimensions):
                        deriv_key = self._higher_order_derivative_key(var, dim1, dim2, order)
                        if deriv_key not in used_derivatives:
                            continue
                        if not order_added:
                            lines.extend(["", f"        # Calculate {self._order_label(order)} derivatives"])
                            order_added = True
                        lines.append(
                            self._higher_order_derivative_line(
                                var,
                                dim1,
                                dim2,
                                order,
                                target_dim_idx,
                            )
                        )
        return lines

    def _order_label(self, order: int) -> str:
        suffixes = {2: "2nd-order", 3: "3rd-order"}
        return suffixes.get(order, f"{order}th-order")

    def _higher_order_derivative_key(self, var: str, dim1: str, dim2: str, order: int) -> str:
        if order == 2:
            return f"{var}_{dim1}{dim2}"
        return f"{var}_{dim1}" + dim2 * (order - 1)

    def _higher_order_derivative_line(
        self,
        var: str,
        dim1: str,
        dim2: str,
        order: int,
        target_dim_idx: int,
    ) -> str:
        if order == 2:
            return (
                f"        d{var}_{dim1}{dim2} = "
                f"self.gradients(d{var}_{dim1}, x_train)[0][..., {target_dim_idx}]"
            )
        prev_deriv = f"d{var}_{dim1}" + dim2 * (order - 2)
        return (
            f"        d{var}_{dim1}{dim2 * (order - 1)} = "
            f"self.gradients({prev_deriv}, x_train)[0][..., {target_dim_idx}]"
        )

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

        normalized = self._replace_derivative_tokens(eq_str)
        if "=" in normalized:
            lhs, rhs = normalized.split("=")
            normalized = f"({lhs}) - ({rhs})"

        try:
            return sp.sympify(normalized)
        except Exception as exc:
            raise ValueError(f"EquationAnalyticalError: {exc}, 原Equation: {normalized}") from exc

    def _find_used_derivatives(self, equations: Union[List[str], Dict[str, List[str]]]) -> Dict[str, int]:
        """
        AnalyzeEquation中Using的Derivatives及其最高阶数
        """
        used_derivatives: Dict[str, int] = {}
        for eq in self._normalize_equations(equations):
            if eq == "0":
                continue
            self._record_used_symbols_for_equation(used_derivatives, eq)
        return used_derivatives

    def _record_used_symbols_for_equation(
        self,
        used_derivatives: Dict[str, int],
        equation: str,
    ) -> None:
        for var in self.vars_list:
            if var in equation:
                used_derivatives[var] = max(used_derivatives.get(var, 0), 0)
            for dim in self.dimensions:
                self._record_matching_derivatives(used_derivatives, equation, var, dim)

    def _record_matching_derivatives(
        self,
        used_derivatives: Dict[str, int],
        equation: str,
        var: str,
        dim: str,
    ) -> None:
        for order in range(1, 6):
            pattern = f"diff({var},{dim})" if order == 1 else f"diff({var},{dim},{order})"
            if pattern in equation:
                self._record_used_derivative(used_derivatives, var, dim, order)

    def generate_pytorch_derivatives(self, equations: Union[List[str], Dict[str, List[str]]]) -> str:
        """
        GeneratePyTorchNeural network的DerivativesCompute代yard
        """
        used_derivatives = self._find_used_derivatives(equations)
        derivatives_code: List[str] = []
        derivatives_code.extend(self._build_variable_extract_lines(used_derivatives))
        derivatives_code.extend(self._build_first_order_derivative_lines(used_derivatives))
        derivatives_code.extend(self._build_higher_order_derivative_lines(used_derivatives))
        return "\n".join(derivatives_code)

    def _convert_equation_to_pytorch(self, eq: str) -> str:
        """
        将Equation字符串Convert为PyTorchformat
        """
        return self._replace_derivative_tokens(eq, prefix="d")

    def generate_unified_operators(self, config_dict: Dict) -> str:
        from .spotter_codegen import build_unified_operators

        return build_unified_operators(self, config_dict)

    def generate_code_for_pytorch_net(
        self,
        equations: Union[List[str], Dict[str, List[str]]],
        output_path: str,
        problem_type: str = "linear_pde",
        config_dict: Optional[Dict] = None,
    ) -> None:
        from .spotter_codegen import compose_pytorch_code

        code = compose_pytorch_code(
            processor=self,
            equations=equations,
            problem_type=problem_type,
            config_dict=config_dict,
        )
        with open(output_path, "w") as output_file:
            output_file.write(code)
        print(f"PyTorchNeural network代yard已GenerateTo: {output_path}")
