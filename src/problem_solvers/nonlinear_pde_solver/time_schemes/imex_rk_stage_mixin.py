from typing import Any, Dict, List, Tuple

import numpy as np


class ImexRKStageMixin:
    def _nonlinear_terms_to_matrix(self, nonlinear_terms, n_points: int) -> np.ndarray:
        """Normalize N_func output to a dense (n_points, n_eqs) array."""
        if nonlinear_terms is None:
            return np.zeros((n_points, self.config.n_eqs))

        if isinstance(nonlinear_terms, np.ndarray):
            if nonlinear_terms.ndim == 2:
                return nonlinear_terms
            if nonlinear_terms.ndim == 1:
                return nonlinear_terms.reshape(n_points, 1)

        matrix = np.zeros((n_points, self.config.n_eqs))
        for eq_idx, values in enumerate(nonlinear_terms):
            if eq_idx >= self.config.n_eqs:
                break
            matrix[:, eq_idx] = np.asarray(values).reshape(-1)[:n_points]
        return matrix

    def time_step(
        self,
        U_n: np.ndarray,
        U_seg: List[np.ndarray],
        dt: float,
        coeffs_n: np.ndarray = None,
        current_time: float = 0.0,
        step: int = 0,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute IMEX-RK(2,2,2) time step"""
        _ = (U_n, coeffs_n)
        self._current_time = current_time
        self.U_seg_n = U_seg.copy()

        U_seg_stage1, coeffs_stage1 = self._solve_imex_stage_U_seg(
            dt=dt,
            stage=1,
            step=step,
            U_n_seg=self.U_seg_n,
        )
        U_seg_stage2, coeffs_stage2 = self._solve_imex_stage_U_seg(
            dt=dt,
            stage=2,
            step=step,
            U_n_seg=self.U_seg_n,
            U_1_seg=U_seg_stage1,
            coeffs_1=coeffs_stage1,
        )
        U_seg_new = self._imex_final_update_U_seg(
            U_n_seg=self.U_seg_n,
            U_1_seg=U_seg_stage1,
            U_2_seg=U_seg_stage2,
            dt=dt,
            coeffs_1=coeffs_stage1,
            coeffs_2=coeffs_stage2,
        )
        U_new = self.fitter.segments_to_global(U_seg_new)
        return U_new, U_seg_new, coeffs_stage2

    def _solve_imex_stage_U_seg(self, **kwargs) -> Tuple[List[np.ndarray], np.ndarray]:
        """SolveIMEX-RKPhase - 直接Using段级SolutionvalueOperation"""
        stage_data = {
            "U_n_seg": kwargs.get("U_n_seg"),
            "U_prev_seg": kwargs.get("U_1_seg", None),
            "U_1_seg": kwargs.get("U_1_seg", None),
            "dt": kwargs.get("dt"),
            "stage": kwargs.get("stage"),
            "operation": "imex_stage",
            "coeffs_n": kwargs.get("coeffs_1", None),
            "coeffs_1": kwargs.get("coeffs_1", None),
            "step": kwargs.get("step"),
        }
        coeffs_stage = self.fitter.fit(**stage_data)
        _, U_stage_segments = self.fitter.construct(self.fitter.data, self.fitter._current_model, coeffs_stage)
        K_stage_segments = self._compute_stage_operators(U_stage_segments, coeffs_stage, kwargs)
        stage_num = kwargs.get("stage")
        setattr(self, f"_K_stage_{stage_num}", K_stage_segments)

        if stage_num == 1:
            explicit_stage1_segments = self._compute_explicit_stage1_terms(U_stage_segments, coeffs_stage)
            setattr(self, "_explicit_stage1_terms", explicit_stage1_segments)

        return U_stage_segments, coeffs_stage

    def _compute_stage_operators(
        self,
        U_stage_segments: List[np.ndarray],
        coeffs_stage: np.ndarray,
        kwargs: dict,
    ) -> List[np.ndarray]:
        """ComputePhaseOperatorsvalue K = L1(U) + L2(U)⊙F(U) + N(U)"""
        K_segments = []
        stage = kwargs.get("stage")
        U_n_seg = kwargs.get("U_n_seg", [])
        U_1_seg = kwargs.get("U_1_seg", [])

        for segment_idx in range(self.fitter.ns):
            U_seg = U_stage_segments[segment_idx]
            K_seg = self._build_segment_operator_contribution(segment_idx, U_seg, coeffs_stage, stage, U_n_seg, U_1_seg)
            K_segments.append(K_seg)
        return K_segments

    def _build_segment_operator_contribution(
        self,
        segment_idx: int,
        U_seg: np.ndarray,
        coeffs_stage: np.ndarray,
        stage: int,
        U_n_seg: List[np.ndarray],
        U_1_seg: List[np.ndarray],
    ) -> np.ndarray:
        K_seg = np.zeros_like(U_seg)
        L1_ops = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_ops = self.fitter._linear_operators[segment_idx].get("L2", None)
        features_list = self.fitter._features[segment_idx]
        F_vals = self.fitter.F_func(features_list, U_seg)

        for eq_idx in range(self.config.n_eqs):
            beta = coeffs_stage[segment_idx, eq_idx, :]
            K_seg[:, eq_idx] += L1_ops[eq_idx] @ beta
            L2_contrib = L2_ops[eq_idx] @ beta
            K_seg[:, eq_idx] += L2_contrib * F_vals[:, eq_idx]

        N_input = U_n_seg[segment_idx] if stage == 1 else (U_1_seg[segment_idx] if U_1_seg else U_seg)
        if self.fitter.has_operator("N"):
            K_seg += self._nonlinear_terms_to_matrix(
                self.fitter.N_func(features_list, N_input),
                n_points=U_seg.shape[0],
            )
        return K_seg

    def _compute_explicit_stage1_terms(
        self,
        U_stage1_segments: List[np.ndarray],
        coeffs_stage1: np.ndarray,
    ) -> List[np.ndarray]:
        """预Computestage1的显式Item [L1 + L2⊙F(U^(1))]β^(1) 用于stage2的RHS"""
        explicit_segments = []
        for segment_idx in range(self.fitter.ns):
            U_seg = U_stage1_segments[segment_idx]
            n_points = U_seg.shape[0]
            ne = self.config.n_eqs
            L1_ops = self.fitter._linear_operators[segment_idx].get("L1", None)
            L2_ops = self.fitter._linear_operators[segment_idx].get("L2", None)
            features_list = self.fitter._features[segment_idx]
            F_vals = (
                self.fitter.F_func(features_list, U_seg)
                if L2_ops is not None and self.fitter.has_operator("F")
                else None
            )
            explicit_seg = np.zeros((n_points, ne))
            for eq_idx in range(ne):
                beta_1 = coeffs_stage1[segment_idx, eq_idx, :]
                if L1_ops is not None:
                    explicit_seg[:, eq_idx] += L1_ops[eq_idx] @ beta_1
                if L2_ops is not None and F_vals is not None:
                    L2_contrib = L2_ops[eq_idx] @ beta_1
                    explicit_seg[:, eq_idx] += L2_contrib * F_vals[:, eq_idx]
            explicit_segments.append(explicit_seg)
        return explicit_segments

    def _imex_final_update_U_seg(
        self,
        U_n_seg: List[np.ndarray],
        dt: float,
        **_kwargs,
    ) -> List[np.ndarray]:
        """IMEX-RKFinalUpdateStep - 直接Using预Compute的OperatorsvalueK"""
        if not hasattr(self, "_K_stage_1") or not hasattr(self, "_K_stage_2"):
            raise RuntimeError("Stage operator values not computed. This is a programming error.")

        U_seg_new = []
        for segment_idx in range(self.fitter.ns):
            U_n_current = U_n_seg[segment_idx]
            K1 = -self._K_stage_1[segment_idx]
            K2 = -self._K_stage_2[segment_idx]
            U_seg_new.append(U_n_current + dt * 0.5 * (K1 + K2))
        return U_seg_new

    def build_stage_jacobian(self, segment_idx: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """BuildIMEX-RKPhase的段Jacobian matrix"""
        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.fitter.dgN
        stage = kwargs.get("stage")
        dt = kwargs.get("dt")
        L1_operators = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)
        features_list = self.fitter._features[segment_idx]
        features = features_list[0]
        V = features_list[0]

        U_n_seg, U_prev_seg = self._resolve_stage_inputs(segment_idx, kwargs)
        F_n = self.fitter.F_func(features_list, U_n_seg)
        F_prev = self.fitter.F_func(features_list, U_prev_seg) if U_prev_seg is not None else None
        N_n = (
            self._nonlinear_terms_to_matrix(
                self.fitter.N_func(features_list, U_n_seg),
                n_points=n_points,
            )
            if self.fitter.has_operator("N")
            else None
        )
        N_prev = (
            self._nonlinear_terms_to_matrix(
                self.fitter.N_func(features_list, U_prev_seg),
                n_points=n_points,
            )
            if U_prev_seg is not None and self.fitter.has_operator("N")
            else None
        )
        beta_prev = self._resolve_previous_coefficients(segment_idx, kwargs)

        rhs = self._build_stage_rhs(
            segment_idx,
            L1_operators=L1_operators,
            L2_operators=L2_operators,
            U_n_seg=U_n_seg,
            U_prev_seg=U_prev_seg,
            F_n=F_n,
            F_prev=F_prev,
            features=features,
            N_n=N_n,
            N_prev=N_prev,
            beta_prev=beta_prev,
            stage=stage,
            dt=dt,
        )

        L_final = np.zeros((ne * n_points, ne * dgN))
        b_final = []
        for eq_idx in range(ne):
            row_start = eq_idx * n_points
            row_end = (eq_idx + 1) * n_points
            col_start = eq_idx * dgN
            col_end = (eq_idx + 1) * dgN
            J_eq = self._build_equation_jacobian(
                V,
                L1_operators[eq_idx],
                L2_operators[eq_idx],
                stage,
                dt,
                eq_idx,
                F_n,
                F_prev,
            )
            L_final[row_start:row_end, col_start:col_end] = J_eq
            b_final.append(rhs[:, eq_idx])
        return L_final, np.concatenate(b_final)

    def _resolve_stage_inputs(self, segment_idx: int, kwargs: dict):
        U_n_seg_list = kwargs.get("U_n_seg", [])
        U_prev_seg_list = kwargs.get("U_prev_seg", kwargs.get("U_1_seg", []))
        U_n_seg = U_n_seg_list[segment_idx]
        U_prev_seg = U_prev_seg_list[segment_idx] if U_prev_seg_list else None
        return U_n_seg, U_prev_seg

    def _resolve_previous_coefficients(self, segment_idx: int, kwargs: dict):
        beta = kwargs.get("coeffs_n", kwargs.get("coeffs_1", None))
        return beta[segment_idx, :, :] if beta is not None else None

    def _build_equation_jacobian(
        self,
        V: np.ndarray,
        L1,
        L2,
        stage: int,
        dt: float,
        eq_idx: int,
        F_n,
        F_prev,
    ) -> np.ndarray:
        J_eq = V.copy()
        if L1 is not None:
            J_eq += self.gamma * dt * L1
        if L2 is not None:
            F_vals = self._stage_F_values(stage, F_n, F_prev)
            if F_vals is not None:
                F_eq = F_vals[:, eq_idx]
                J_eq += self.gamma * dt * np.diag(F_eq) @ L2
        return J_eq

    def _stage_F_values(self, stage: int, F_n, F_prev):
        if stage == 1:
            return F_n
        if stage == 2:
            return F_prev
        raise ValueError(f"Invalid stage: {stage}")

    def _build_stage_rhs(self, _segment_idx: int, **kwargs) -> np.ndarray:
        """BuildIMEX-RKPhase的RightEndVector"""
        dt = kwargs.get("dt")
        stage = kwargs.get("stage")
        U_n_seg = kwargs.get("U_n_seg")
        N_n = kwargs.get("N_n")
        N_prev = kwargs.get("N_prev")
        n_points = U_n_seg.shape[0]
        rhs = np.zeros((n_points, self.config.n_eqs))

        if stage == 1:
            rhs[:, :] = U_n_seg
            if N_n is not None:
                rhs -= self.gamma * dt * N_n
            return rhs

        rhs[:, :] = U_n_seg
        if hasattr(self, "_explicit_stage1_terms") and self._explicit_stage1_terms:
            explicit_seg = self._explicit_stage1_terms[_segment_idx]
            rhs -= dt * (1 - 2 * self.gamma) * explicit_seg
        if N_prev is not None:
            rhs -= dt * (1 - self.gamma) * N_prev
        return rhs

    def estimate_stable_dt(self, _U_current: np.ndarray, safety_factor: float = 0.8) -> float:
        """估算Stable的Time step size"""
        if not self.fitter._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        max_eigenvalue = 0.0
        max_eigenvalue = self._estimate_linear_operator_radius(max_eigenvalue)
        if self.fitter.has_operator("L2") and self.fitter.has_operator("F"):
            max_eigenvalue = max(max_eigenvalue, 1.0)
        if self.fitter.has_operator("N"):
            max_eigenvalue = max(max_eigenvalue, 1.0)
        if max_eigenvalue <= 0:
            return 0.1
        return safety_factor * 1.5 / max_eigenvalue

    def _estimate_linear_operator_radius(self, current_max: float) -> float:
        if not self.fitter.has_operator("L1"):
            return current_max
        max_eigenvalue = current_max
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

    def get_scheme_info(self) -> Dict[str, Any]:
        """GetIMEX-RK(2,2,2)Timeformatinformation"""
        return {
            "method": "IMEX-RK(2,2,2)",
            "stages": 2,
            "order": 2,
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
        """VerificationOperatorsConfigurationYesNoCorrect"""
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

    def get_butcher_tableau(self) -> Dict[str, np.ndarray]:
        """GetIMEX-RK(2,2,2) Butcher表"""
        return {
            "A_implicit": self.A_imp,
            "A_explicit": self.A_exp,
            "b_weights": self.b,
            "gamma": self.gamma,
        }
