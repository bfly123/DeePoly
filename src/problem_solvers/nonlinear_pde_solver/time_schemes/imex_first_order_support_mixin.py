from typing import Any, Dict

import numpy as np


class ImexFirstOrderSupportMixin:
    def get_scheme_info(self) -> Dict[str, Any]:
        return {
            "method": "IMEX First-order (Backward Euler)",
            "stages": 1,
            "order": 1,
            "stability": "A-stable for stiff terms, explicit for non-stiff terms",
            "gamma": self.gamma,
            "operators": {
                "L1": "Implicit linear operator (e.g., diffusion)",
                "L2": "Implicit linear operator with nonlinear coefficient",
                "N": "Explicit nonlinear operator (e.g., reaction)",
                "F": "Nonlinear coefficient function",
            },
            "equation_form": "dU/dt = L1(U) + N(U) + L2(U)*F(U)",
        }

    def validate_operators(self) -> Dict[str, bool]:
        validation = {
            "L1_exists": self.fitter.has_operator("L1"),
            "L2_exists": self.fitter.has_operator("L2"),
            "N_exists": self.fitter.has_operator("N"),
            "F_exists": self.fitter.has_operator("F"),
            "precompiled": self.fitter._precompiled,
        }
        validation["imex_ready"] = validation["precompiled"] and (
            validation["L1_exists"] or validation["L2_exists"]
        )
        return validation

    def estimate_stable_dt(self, _U_current: np.ndarray, safety_factor: float = 0.8) -> float:
        if not self.fitter._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")
        max_eigenvalue = 0.0
        if self.fitter.has_operator("L1"):
            max_eigenvalue = max(max_eigenvalue, self._estimate_l1_spectral_radius())
        if self.fitter.has_operator("L2") and self.fitter.has_operator("F"):
            max_eigenvalue = max(max_eigenvalue, 1.0)
        if self.fitter.has_operator("N"):
            max_eigenvalue = max(max_eigenvalue, 1.0)
        if max_eigenvalue <= 0:
            return 0.1
        return safety_factor * 2.0 / max_eigenvalue

    def _estimate_l1_spectral_radius(self) -> float:
        max_eigenvalue = 0.0
        for segment_idx in range(self.fitter.ns):
            L1_op = self.fitter._linear_operators[segment_idx].get("L1", None)
            if L1_op is None:
                continue
            try:
                eigenvals = np.linalg.eigvals(L1_op)
                max_eigenvalue = max(max_eigenvalue, np.max(np.real(eigenvals)))
            except Exception:
                max_eigenvalue = max(max_eigenvalue, np.max(np.abs(L1_op)))
        return max_eigenvalue

    def get_butcher_tableau(self) -> Dict[str, np.ndarray]:
        return {
            "A_implicit": self.A_imp,
            "A_explicit": self.A_exp,
            "b_weights": self.b,
            "gamma": self.gamma,
        }
