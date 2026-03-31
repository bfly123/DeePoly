import re


class LossCodeGeneratorCodegenMixin:
    _DIFF_PATTERN = r"diff\((\w+),(\w+)(?:,(\d+))?\)"

    def _parse_derivatives(self, equation: str):
        """Parse derivatives from a PDE equation string."""
        derivatives = {}
        for match in re.finditer(self._DIFF_PATTERN, equation):
            var, dim, order = match.groups()
            order = int(order) if order else 1
            dims = derivatives.setdefault(var, [])
            for derivative_order in range(1, order + 1):
                token = dim * derivative_order
                if token not in dims:
                    dims.append(token)
        return derivatives

    def _generate_derivatives_code(self, equations):
        """Generate derivative extraction code for all referenced variables."""
        used_derivatives = self._collect_used_derivatives(equations)
        code_lines = self._build_constant_lines()
        code_lines.extend(self._build_output_lines())
        code_lines.extend(self._build_first_order_lines(used_derivatives))
        second_order_lines = self._build_second_order_lines(used_derivatives)
        if second_order_lines:
            code_lines.extend(["", "# Calculate second-order derivatives"])
            code_lines.extend(second_order_lines)
        return "\n".join(code_lines)

    def _collect_used_derivatives(self, equations):
        used_derivatives = {}
        for equation in equations:
            derivatives = self._parse_derivatives(equation)
            for var, dims in derivatives.items():
                used_derivatives.setdefault(var, []).extend(dims)
        for var, dims in used_derivatives.items():
            used_derivatives[var] = list(dict.fromkeys(dims))
        return used_derivatives

    def _build_constant_lines(self):
        code_lines = []
        if self.const_list:
            code_lines.append("# Extract constants")
            for const_dict in self.const_list:
                for name, value in const_dict.items():
                    code_lines.append(f"{name} = {value}")
            code_lines.append("")
        return code_lines

    def _build_output_lines(self):
        code_lines = ["# Extract physical quantities from output"]
        for index, var in enumerate(self.vars_list):
            code_lines.append(f"{var} = output[..., {index}]")
        code_lines.extend(["", "# Calculate derivatives in each direction"])
        return code_lines

    def _build_first_order_lines(self, used_derivatives):
        code_lines = []
        for var, dims in used_derivatives.items():
            for dim in dims:
                if len(dim) != 1:
                    continue
                dim_idx = self.dimensions.index(dim)
                code_lines.append(
                    f"d{var}_{dim} = self.gradients({var}, x_train)[0][..., {dim_idx}]"
                )
        return code_lines

    def _build_second_order_lines(self, used_derivatives):
        code_lines = []
        for var, dims in used_derivatives.items():
            for dim in dims:
                if len(dim) != 2:
                    continue
                dim2_idx = self.dimensions.index(dim[1])
                first_deriv = f"d{var}_{dim[0]}"
                code_lines.append(
                    f"d{var}_{dim} = self.gradients({first_deriv}, x_train)[0][..., {dim2_idx}]"
                )
        return code_lines

    def _generate_equations_code(self, equations, is_nonlinear: bool = False) -> str:
        """Generate equation evaluation code."""
        eq_exprs = [self._replace_diff_calls(eq) for eq in equations]
        if is_nonlinear:
            code_lines = ["# Add nonlinear terms"]
            for index, expr in enumerate(eq_exprs):
                code_lines.append(f"eq[{index}] = eq[{index}] + {expr}")
            return "\n".join(code_lines)
        code_lines = ["# Compute equations as a list", f"eq = [{', '.join(eq_exprs)}]"]
        return "\n".join(code_lines)

    def _replace_diff_calls(self, equation: str) -> str:
        def replace_diff(match):
            var, dim, order = match.groups()
            order = int(order) if order else 1
            if order == 1:
                return f"d{var}_{dim}"
            return f"d{var}_{dim * order}"

        return re.sub(self._DIFF_PATTERN, replace_diff, equation)

    @staticmethod
    def _generate_loss_code(num_equations: int) -> str:
        """Generate PDE loss accumulation code."""
        if num_equations == 1:
            return "pde_loss = torch.mean((eq[0] - source[0]) ** 2)"
        return (
            "pde_loss = torch.mean("
            f"sum((eq[i] - source[i]) ** 2 for i in range({num_equations}))"
            ")"
        )
