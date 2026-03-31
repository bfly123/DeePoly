from typing import Dict, List, Tuple

import numpy as np
import sympy as sp


class PseudoTimeFitterFrozenNonlinearMixin:
    def _build_frozen_nonlinear_block(
        self,
        eq_idx: int,
        features_list: List[np.ndarray],
        U_seg: np.ndarray,
        ne: int,
        dgN: int,
    ) -> np.ndarray:
        if getattr(self.config, "frozen_nonlinear_jacobian", "none") != "transport":
            return np.zeros((features_list[0].shape[0], ne * dgN))

        frozen_specs = self._get_frozen_nonlinear_specs()
        if eq_idx not in frozen_specs:
            return np.zeros((features_list[0].shape[0], ne * dgN))

        block = np.zeros((features_list[0].shape[0], ne * dgN))
        for coeff, advector_var_idx, target_var_idx, deriv_idx in frozen_specs[eq_idx]:
            start = target_var_idx * dgN
            stop = start + dgN
            advection_speed = U_seg[:, advector_var_idx].reshape(-1)
            block[:, start:stop] += coeff * (np.diag(advection_speed) @ features_list[deriv_idx])
        return block

    def _get_frozen_nonlinear_specs(self) -> Dict[int, List[Tuple[float, int, int, int]]]:
        if hasattr(self, "_frozen_nonlinear_specs"):
            return self._frozen_nonlinear_specs

        specs: Dict[int, List[Tuple[float, int, int, int]]] = {}
        if not self._uses_reference_stable_formulation():
            self._frozen_nonlinear_specs = specs
            return specs

        symbol_table = {
            name: sp.Symbol(name)
            for name in getattr(self, "all_derivatives", {})
        }
        nonlinear_terms = self.operator_terms.get("N", []) if hasattr(self, "operator_terms") else []
        all_derivatives = getattr(self, "all_derivatives", {})

        for eq_idx, term in enumerate(nonlinear_terms):
            expr = term.get("symbolic_expr", "0") if isinstance(term, dict) else "0"
            if not expr or expr == "0":
                continue

            parsed_specs: List[Tuple[float, int, int, int]] = []
            sym_expr = sp.expand(sp.sympify(expr, locals=symbol_table))
            for sym_term in sp.Add.make_args(sym_expr):
                coeff_value, factors = sym_term.as_coeff_mul()
                if len(factors) != 2:
                    continue

                factor_names = [str(factor) for factor in factors]
                factor_info = [all_derivatives.get(name) for name in factor_names]
                if any(info is None for info in factor_info):
                    continue

                left_info, right_info = factor_info
                if left_info[1] == 0 and right_info[1] != 0:
                    advector_info, target_info = left_info, right_info
                elif right_info[1] == 0 and left_info[1] != 0:
                    advector_info, target_info = right_info, left_info
                else:
                    continue

                parsed_specs.append(
                    (
                        float(coeff_value),
                        int(advector_info[0]),
                        int(target_info[0]),
                        int(target_info[1]),
                    )
                )

            if parsed_specs:
                specs[eq_idx] = parsed_specs

        self._frozen_nonlinear_specs = specs
        return specs
