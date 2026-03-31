from typing import Dict, Tuple

import numpy as np


class PseudoTimeFitterResidualMixin:
    def _initialize_coefficients(self) -> np.ndarray:
        """Initialize coefficient array with zeros plus a tiny perturbation."""
        coeffs = np.zeros((self.ns, self.config.n_eqs, self.dgN))
        coeffs += np.random.randn(*coeffs.shape) * self.COEFF_PERTURBATION
        return coeffs

    def compute_residual(self, U: np.ndarray) -> np.ndarray:
        """Compute residual R(U) = L1(U) + L2(U)·F(U) + N(U) - S."""
        self._validate_residual_inputs()
        U_seg = self.global_to_segments(U)
        residual = np.zeros_like(U)
        for seg_idx, segment_values in enumerate(U_seg):
            start_idx, end_idx = self._get_segment_global_indices(seg_idx)
            residual[start_idx:end_idx] = self._compute_segment_residual(seg_idx, segment_values)
        return residual

    def _validate_residual_inputs(self) -> None:
        if not self._precompiled or not hasattr(self, "_coeffs_current") or self._coeffs_current is None:
            raise RuntimeError("Operators not precompiled or coefficients not available")

    def _compute_segment_residual(self, seg_idx: int, U_seg: np.ndarray) -> np.ndarray:
        features = self._features[seg_idx]
        coeffs = self._coeffs_current[seg_idx]
        L1_ops = self._linear_operators[seg_idx].get("L1")
        L2_ops = self._linear_operators[seg_idx].get("L2")
        F_val = self.F_func(features, U_seg)
        N_val = self.N_func(features, U_seg)
        S_val = self._get_source_term(seg_idx)

        residual_seg = np.zeros_like(U_seg)
        for eq_idx in range(self.config.n_eqs):
            residual_seg[:, eq_idx] = self._compute_equation_residual(
                eq_idx,
                coeffs,
                L1_ops,
                L2_ops,
                F_val,
                N_val,
                S_val,
            )
        return residual_seg

    def _compute_equation_residual(
        self,
        eq_idx: int,
        coeffs: np.ndarray,
        L1_ops,
        L2_ops,
        F_val,
        N_val,
        S_val,
    ) -> np.ndarray:
        residual = np.array(N_val[eq_idx], copy=True) - S_val[:, eq_idx]
        if L1_ops is not None and eq_idx < len(L1_ops):
            residual += L1_ops[eq_idx] @ coeffs[eq_idx]
        if L2_ops is not None and eq_idx < len(L2_ops):
            residual += (L2_ops[eq_idx] @ coeffs[eq_idx]) * F_val[:, eq_idx]
        return residual

    def _cache_segment_indices(self) -> None:
        """Pre-compute and cache segment global indices."""
        self._segment_start_indices = []
        self._segment_end_indices = []
        cumsum = 0
        for seg in self.data["x_segments_norm"]:
            self._segment_start_indices.append(cumsum)
            cumsum += len(seg)
            self._segment_end_indices.append(cumsum)

    def _get_segment_global_indices(self, segment_idx: int) -> Tuple[int, int]:
        """Get global array indices for a segment."""
        return self._segment_start_indices[segment_idx], self._segment_end_indices[segment_idx]

    def _compute_convergence_metrics(self, U_new: np.ndarray, U_current: np.ndarray) -> Dict[str, float]:
        """Compute composite convergence metrics for pseudo-time stepping."""
        delta_U = U_new - U_current
        solution_norm = float(np.linalg.norm(U_new))
        current_norm = float(np.linalg.norm(U_current))
        relative_l2_update = float(np.linalg.norm(delta_U) / max(solution_norm, current_norm, 1.0))

        n_eqs = U_new.shape[1] if U_new.ndim > 1 else 1
        primary_var = int(getattr(self, "convergence_primary_var", 0))
        primary_var = max(0, min(primary_var, n_eqs - 1))

        if U_new.ndim == 1:
            primary_delta = delta_U
            primary_new = U_new
            primary_current = U_current
        else:
            primary_delta = delta_U[:, primary_var]
            primary_new = U_new[:, primary_var]
            primary_current = U_current[:, primary_var]

        primary_solution_norm = float(np.linalg.norm(primary_new))
        primary_current_norm = float(np.linalg.norm(primary_current))
        primary_relative_update = float(
            np.linalg.norm(primary_delta) / max(primary_solution_norm, primary_current_norm, 1.0)
        )
        primary_max_abs_update = float(np.max(np.abs(primary_delta))) if primary_delta.size else 0.0
        max_abs_update = float(np.max(np.abs(delta_U))) if delta_U.size else 0.0

        return {
            "solution_norm": solution_norm,
            "relative_l2_update": relative_l2_update,
            "primary_var": primary_var,
            "primary_relative_update": primary_relative_update,
            "primary_max_abs_update": primary_max_abs_update,
            "max_abs_update": max_abs_update,
        }

    def _compute_residual_norm(self, U_new: np.ndarray, U_current: np.ndarray) -> Tuple[float, float]:
        """Compute normalized residual and solution norm."""
        metrics = self._compute_convergence_metrics(U_new, U_current)
        return metrics["relative_l2_update"], metrics["solution_norm"]

    def _update_convergence_history(
        self,
        residual: float,
        dt: float,
        solution_norm: float,
        metrics: Dict[str, float] | None = None,
    ) -> None:
        """Update convergence history lists."""
        self.residual_history.append(residual)
        self.dt_history.append(dt)
        self.solution_norm_history.append(solution_norm)
        if metrics is None:
            return
        self.primary_update_history.append(metrics["primary_max_abs_update"])
        self.primary_relative_update_history.append(metrics["primary_relative_update"])
        self.max_abs_update_history.append(metrics["max_abs_update"])
