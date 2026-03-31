from typing import Dict, List, Tuple

import numpy as np
import sympy as sp


class PseudoTimeFitterNonlinearLinearizationMixin:
    def _uses_newton_linearization(self) -> bool:
        return str(getattr(self.config, "nonlinear_linearization", "picard")).lower() == "newton"

    def _build_nonlinear_linearization_block(
        self,
        eq_idx: int,
        features: np.ndarray,
        U_seg: np.ndarray,
        ne: int,
        dgN: int,
    ) -> np.ndarray:
        if not self._uses_newton_linearization():
            return np.zeros((features.shape[0], ne * dgN))

        eq_specs = self._get_newton_nonlinear_specs().get(eq_idx, [])
        if not eq_specs:
            return np.zeros((features.shape[0], ne * dgN))

        block = np.zeros((features.shape[0], ne * dgN))
        for target_var_idx, derivative_func in eq_specs:
            coeff_values = self._evaluate_newton_diagonal(derivative_func, U_seg, features.shape[0])
            start = target_var_idx * dgN
            stop = start + dgN
            block[:, start:stop] += coeff_values[:, None] * features
        return block

    def _build_nonlinear_linearization_rhs_correction(
        self,
        eq_idx: int,
        U_seg: np.ndarray,
    ) -> np.ndarray:
        if not self._uses_newton_linearization():
            return np.zeros(U_seg.shape[0], dtype=np.float64)

        eq_specs = self._get_newton_nonlinear_specs().get(eq_idx, [])
        if not eq_specs:
            return np.zeros(U_seg.shape[0], dtype=np.float64)

        correction = np.zeros(U_seg.shape[0], dtype=np.float64)
        for target_var_idx, derivative_func in eq_specs:
            coeff_values = self._evaluate_newton_diagonal(derivative_func, U_seg, U_seg.shape[0])
            correction += coeff_values * U_seg[:, target_var_idx]
        return correction

    def _evaluate_newton_diagonal(
        self,
        derivative_func,
        U_seg: np.ndarray,
        n_points: int,
    ) -> np.ndarray:
        arg_var_indices = getattr(self, "_newton_arg_var_indices", [])
        args = [U_seg[:, var_idx] for var_idx in arg_var_indices]
        values = derivative_func(*args) if args else derivative_func()
        values_arr = np.asarray(values, dtype=np.float64)
        if values_arr.ndim == 0:
            return np.full(n_points, float(values_arr), dtype=np.float64)

        values_arr = values_arr.reshape(-1)
        if values_arr.size == 1:
            return np.full(n_points, float(values_arr[0]), dtype=np.float64)
        if values_arr.size != n_points:
            raise ValueError(
                "Newton linearization produced incompatible diagonal size: "
                f"expected {n_points}, got {values_arr.size}"
            )
        return values_arr

    def _get_newton_nonlinear_specs(self) -> Dict[int, List[Tuple[int, object]]]:
        if hasattr(self, "_newton_nonlinear_specs"):
            return self._newton_nonlinear_specs

        specs: Dict[int, List[Tuple[int, object]]] = {}
        warnings: List[str] = []
        nonlinear_terms = self.operator_terms.get("N", []) if hasattr(self, "operator_terms") else []
        all_derivatives = getattr(self, "all_derivatives", {}) or {}

        symbol_table = {name: sp.Symbol(name) for name in all_derivatives}
        symbol_table.update(self._get_sympy_constants())
        symbol_table.setdefault("pi", sp.pi)

        zero_order_symbols: Dict[int, sp.Symbol] = {}
        for name, (var_idx, deriv_idx) in all_derivatives.items():
            if int(deriv_idx) != 0:
                continue
            zero_order_symbols.setdefault(int(var_idx), symbol_table[name])

        arg_var_indices = sorted(zero_order_symbols)
        arg_symbols = [zero_order_symbols[var_idx] for var_idx in arg_var_indices]

        for eq_idx, term in enumerate(nonlinear_terms):
            expr = term.get("symbolic_expr", "0") if isinstance(term, dict) else "0"
            if not expr or expr == "0":
                continue

            sym_expr = sp.expand(sp.sympify(expr, locals=symbol_table))
            unsupported_symbols = []
            for symbol in sym_expr.free_symbols:
                symbol_name = str(symbol)
                deriv_info = all_derivatives.get(symbol_name)
                if deriv_info is not None and int(deriv_info[1]) != 0:
                    unsupported_symbols.append(symbol_name)

            if unsupported_symbols:
                warnings.append(
                    "Pseudo-time Newton skipped for equation "
                    f"{eq_idx}: nonlinear term depends on derivatives {sorted(set(unsupported_symbols))}"
                )
                continue

            eq_specs: List[Tuple[int, object]] = []
            for var_idx in arg_var_indices:
                derivative_expr = sp.simplify(sp.diff(sym_expr, zero_order_symbols[var_idx]))
                if derivative_expr == 0:
                    continue
                eq_specs.append(
                    (
                        var_idx,
                        sp.lambdify(arg_symbols, derivative_expr, modules="numpy"),
                    )
                )

            if eq_specs:
                specs[eq_idx] = eq_specs

        self._newton_arg_var_indices = arg_var_indices
        self._newton_nonlinear_specs = specs
        self._newton_nonlinear_warnings = warnings
        self._emit_newton_warnings()
        return specs

    def _emit_newton_warnings(self) -> None:
        if getattr(self, "_newton_warnings_emitted", False):
            return
        for warning in getattr(self, "_newton_nonlinear_warnings", []):
            print(f"Warning: {warning}")
        self._newton_warnings_emitted = True

    def _get_sympy_constants(self) -> Dict[str, float]:
        constants = getattr(self.config, "constants", None)
        if not constants and hasattr(self, "_flatten_const_list"):
            constants = self._flatten_const_list(getattr(self.config, "const_list", []))
        return {
            str(name): float(value)
            for name, value in (constants or {}).items()
            if np.isscalar(value)
        }
