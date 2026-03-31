import functools

from typing import Dict, List, Tuple

import numpy as np
import sympy as sp


class NonlinearEquationCodegenMixin:
    def parse_equation(self, eq_str: str) -> sp.Expr:
        if eq_str == "0":
            return sp.Integer(0)

        for var in self.vars_list:
            for dim in self.dimensions:
                eq_str = eq_str.replace(f"diff({var},{dim})", f"{var}_{dim}")

        if "=" in eq_str:
            lhs, rhs = eq_str.split("=")
            eq_str = f"({lhs}) - ({rhs})"

        try:
            return sp.sympify(eq_str)
        except Exception as exc:
            raise ValueError(f"方程解析错误: {str(exc)}, 原方程: {eq_str}") from exc

    def _format_multiplication(self, factors: List[str]) -> str:
        if len(factors) == 2:
            return f"np.multiply({factors[0]}, {factors[1]})"
        return "functools.reduce(np.multiply, [" + ", ".join(factors) + "])"

    def generate_L_terms(self, equations: Tuple[str, ...]) -> str:
        l_terms = []

        for i, eq in enumerate(equations):
            if eq == "0":
                l_terms.append(f"L[{i}] = np.zeros((n_points, 1))")
                continue

            expr = self.parse_equation(eq)
            terms = []
            for term in sp.Add.make_args(expr):
                factors = sp.Mul.make_args(term)
                if len(factors) > 1:
                    terms.append(self._format_multiplication([str(f) for f in factors]))
                else:
                    terms.append(str(term))

            if terms:
                l_terms.append(f"L[{i}] = " + " + ".join(terms))
            else:
                l_terms.append(f"L[{i}] = np.zeros((n_points, 1))")

        return "\n".join(l_terms)

    def generate_dL_terms(self, equations: Tuple[str, ...]) -> str:
        dl_terms = []

        for i, eq in enumerate(equations):
            expr = self.parse_equation(eq)
            if expr == 0:
                dl_terms.append(f"dL[{i}] = np.zeros((n_points, ne * self.dgN))")
                continue

            jac_terms = []
            for var in self.vars_list:
                var_sym = sp.Symbol(var)
                var_deriv = sp.diff(expr, var_sym)
                if var_deriv != 0:
                    jac_terms.append(
                        f"np.diag(({str(var_deriv)}).flatten()) @ {var.upper()}"
                    )

                for dim in self.dimensions:
                    deriv_sym = self.derivatives[f"{var}_{dim}"]
                    deriv_coeff = sp.diff(expr, deriv_sym)
                    if deriv_coeff != 0:
                        jac_terms.append(
                            "np.diag(("
                            f"{str(deriv_coeff)}"
                            f").flatten()) @ {var.upper()}_{dim}"
                        )

            if jac_terms:
                dl_terms.append(f"dL[{i}] = " + " + ".join(jac_terms))
            else:
                dl_terms.append(f"dL[{i}] = np.zeros((n_points, ne * self.dgN))")

        return "\n".join(dl_terms)

    def _find_used_variables(self, equations: Tuple[str, ...]) -> Dict[str, bool]:
        used_vars = {var: False for var in self.vars_list}
        used_derivatives = {
            f"{var}_{dim}": False
            for var in self.vars_list
            for dim in self.dimensions
        }

        for eq in equations:
            if eq == "0":
                continue

            expr = self.parse_equation(eq)
            for var in self.vars_list:
                var_sym = sp.Symbol(var)
                if expr.has(var_sym):
                    used_vars[var] = True

            for var in self.vars_list:
                for dim in self.dimensions:
                    deriv_name = f"{var}_{dim}"
                    deriv_sym = self.derivatives[deriv_name]
                    if expr.has(deriv_sym):
                        used_derivatives[deriv_name] = True
                        used_vars[var] = True

        return {**used_vars, **used_derivatives}

    def generate_variable_declarations(
        self,
        equations: Tuple[str, ...],
        vars_list: List[str],
        dimensions: List[str],
    ) -> str:
        used_vars = self._find_used_variables(equations)
        declarations = []

        for var in vars_list:
            if used_vars.get(var, False):
                declarations.append(f'{var.upper()} = variables["{var.upper()}"][segment_idx]')

        for var in vars_list:
            for dim in dimensions:
                deriv_name = f"{var}_{dim}"
                if used_vars.get(deriv_name, False):
                    declarations.append(
                        f'{var.upper()}_{dim} = variables["{var.upper()}_{dim}"][segment_idx]'
                    )

        for var in vars_list:
            if used_vars.get(var, False):
                declarations.append(f"{var} = {var.upper()} @ x_slice")
                for dim in dimensions:
                    deriv_name = f"{var}_{dim}"
                    if used_vars.get(deriv_name, False):
                        declarations.append(f"{var}_{dim} = {var.upper()}_{dim} @ x_slice")

        return "\n".join(" " * 8 + line for line in declarations)

    def generate_code_for_hybrid_fitter(
        self,
        equations: Tuple[str, ...],
        output_path: str,
    ) -> None:
        try:
            var_declarations = self.generate_variable_declarations(
                equations,
                self.vars_list,
                self.dimensions,
            )
            l_code = self.generate_L_terms(equations)
            dl_code = self.generate_dL_terms(equations)

            indent = " " * 8
            l_code = "\n".join(indent + line for line in l_code.split("\n"))
            dl_code = "\n".join(indent + line for line in dl_code.split("\n"))

            code = f"""#-----begin auto code-----
{var_declarations}

{l_code}
{dl_code}
#-----end auto code-----
"""
            with open(output_path, "w") as handle:
                handle.write(code)

            print(f"代码已生成到: {output_path}")
        except Exception as exc:
            print(f"生成代码时出错: {str(exc)}")
            raise
