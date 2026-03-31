from typing import Any, Dict

import numpy as np


class OneStepPredictorSupportMixin:
    def build_stage_jacobian(self, segment_idx: int, **kwargs):
        """Build the predictor Jacobian; the corrector is explicit."""
        operation = kwargs.get("operation", "onestep_predictor")
        dt = kwargs.get("dt")
        if operation != "onestep_predictor":
            raise ValueError(f"Only onestep_predictor needs Jacobian. Got: {operation}")
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("dt", None)
        return self._build_predictor_jacobian(segment_idx, dt, **kwargs_copy)

    def _build_predictor_jacobian(self, segment_idx: int, dt: float, **kwargs):
        """Build the linear system for the implicit predictor step."""
        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.fitter.dgN
        L1_ops = self.fitter._linear_operators[segment_idx].get("L1")
        L2_ops = self.fitter._linear_operators[segment_idx].get("L2")
        features_list = self.fitter._features[segment_idx]
        features = features_list[0]
        U_n_seg = self._get_segment_solution(kwargs, segment_idx)
        F_n = self.fitter.F_func(features, U_n_seg)
        N_n = self.fitter.N_func(features, U_n_seg)

        A_matrix = np.zeros((ne * n_points, ne * dgN), dtype=np.float64)
        b_vector = []
        for eq_idx in range(ne):
            row_start, row_end = eq_idx * n_points, (eq_idx + 1) * n_points
            col_start, col_end = eq_idx * dgN, (eq_idx + 1) * dgN
            J_eq = features.copy()
            J_eq += dt * L1_ops[eq_idx]
            J_eq += dt * np.diag(F_n[:, eq_idx]) @ L2_ops[eq_idx]
            A_matrix[row_start:row_end, col_start:col_end] = J_eq
            rhs_eq = U_n_seg[:, eq_idx].copy() - dt * N_n[eq_idx]
            b_vector.append(rhs_eq)
        return A_matrix, np.concatenate(b_vector)

    @staticmethod
    def _get_segment_solution(kwargs, segment_idx: int):
        U_n_seg_list = kwargs.get("U_n_seg", [])
        if len(U_n_seg_list) <= segment_idx:
            raise ValueError(f"U_n_seg_list does not contain segment {segment_idx}")
        return U_n_seg_list[segment_idx]

    def get_scheme_info(self) -> Dict[str, Any]:
        """Return the scheme metadata."""
        return {
            "name": "OneStepPredictor",
            "type": "predictor-corrector",
            "method": "One-Step Predictor-Corrector",
            "order": self.order,
            "implicit": "predictor only",
            "explicit": "corrector only",
            "stages": 2,
            "conditional_corrector": True,
            "operator_reuse": True,
            "first_step_reduction": True,
            "equation_form": "∂u/∂t + L1(u) + L2(u)⋅F(u) + N(u) = 0",
            "description": "One-step predictor-corrector scheme，operator values预Compute和Time stepBetween复用",
        }

    @staticmethod
    def validate_operators() -> Dict[str, bool]:
        """One-step predictor assumes all operators are available."""
        return {
            "L1_exists": True,
            "L2_exists": True,
            "F_exists": True,
            "N_exists": True,
            "onestep_ready": True,
        }

    def estimate_stable_dt(self, u_current: np.ndarray, safety_factor: float = 0.8) -> float:
        """Estimate a conservative stable time step."""
        if not hasattr(self.fitter, "data") or self.fitter.data is None:
            return 0.01
        dx_min = self._estimate_min_spacing()
        dt_diffusion = self._estimate_diffusion_dt(dx_min)
        dt_nonlinear = self._estimate_nonlinear_dt(u_current)
        dt_stable = min(dt_diffusion, dt_nonlinear) * safety_factor
        return max(1e-6, min(dt_stable, 0.1))

    def _estimate_min_spacing(self) -> float:
        try:
            x_data = self.fitter.data.get("x", np.array([0.0, 1.0]))
            x_flat = x_data.flatten()
            if x_flat.size > 1:
                return np.min(np.diff(np.sort(x_flat)))
            return 0.1
        except Exception:
            return 0.1

    @staticmethod
    def _estimate_diffusion_dt(dx_min: float) -> float:
        diffusion_coeff = 0.0001
        if diffusion_coeff <= 0:
            return float("inf")
        return dx_min**2 / (2 * diffusion_coeff)

    @staticmethod
    def _estimate_nonlinear_dt(u_current: np.ndarray) -> float:
        if len(u_current) == 0:
            return float("inf")
        u_max = np.max(np.abs(u_current))
        if u_max <= 0:
            return float("inf")
        return 0.01 / (u_max**2 + 1e-6)
