from typing import List, Tuple

import numpy as np

from .pseudotime_fitter_formulation_mixin import PseudoTimeFitterFormulationMixin
from .pseudotime_fitter_frozen_nonlinear_mixin import (
    PseudoTimeFitterFrozenNonlinearMixin,
)
from .pseudotime_fitter_nonlinear_linearization_mixin import (
    PseudoTimeFitterNonlinearLinearizationMixin,
)


class PseudoTimeFitterJacobianMixin(
    PseudoTimeFitterFormulationMixin,
    PseudoTimeFitterFrozenNonlinearMixin,
    PseudoTimeFitterNonlinearLinearizationMixin,
):
    def _build_segment_jacobian(
        self,
        segment_idx: int,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the implicit Euler Jacobian for a single segment."""
        U_n_seg_list, dt = self._extract_segment_jacobian_inputs(kwargs)
        U_seg = U_n_seg_list[segment_idx]
        n_points = len(self.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.dgN

        L1_operators = self._linear_operators[segment_idx].get("L1")
        L2_operators = self._linear_operators[segment_idx].get("L2")
        features_list = self._features[segment_idx]
        features = features_list[0]
        reference_stable = self._uses_reference_stable_formulation()
        F_n, N_n = self._evaluate_segment_operators(
            features_list,
            U_seg,
            n_points,
            ne,
            reference_stable,
        )
        source_term = self._get_source_term(segment_idx)
        steady_state_mode = getattr(self.config, "steady_state_mode", False)

        system_matrix = np.zeros((ne * n_points, ne * dgN))
        rhs_vectors = []
        for eq_idx in range(ne):
            row_start = eq_idx * n_points
            row_end = (eq_idx + 1) * n_points
            col_start = eq_idx * dgN
            col_end = (eq_idx + 1) * dgN

            jacobian_block = self._build_equation_jacobian(
                features_list,
                U_seg,
                features,
                dt,
                L1_operators[eq_idx],
                L2_operators[eq_idx],
                F_n[:, eq_idx],
                steady_state_mode,
                reference_stable,
                eq_idx,
                ne,
                dgN,
            )
            system_matrix[row_start:row_end, :] = jacobian_block
            rhs_vectors.append(
                self._build_equation_rhs(
                    U_seg,
                    dt,
                    source_term[:, eq_idx],
                    N_n[:, eq_idx],
                    steady_state_mode,
                    reference_stable,
                    eq_idx,
                )
            )

        return system_matrix, np.concatenate(rhs_vectors)

    def _extract_segment_jacobian_inputs(self, kwargs) -> Tuple[List[np.ndarray], float]:
        U_n_seg_list = kwargs.get("U_n_seg")
        dt = kwargs.get("dt")
        if U_n_seg_list is None:
            raise ValueError("U_n_seg must be provided in kwargs")
        if dt is None:
            raise ValueError("dt must be provided in kwargs for pseudo-time stepping")
        return U_n_seg_list, dt

    def _evaluate_segment_operators(
        self,
        features,
        U_seg: np.ndarray,
        n_points: int,
        ne: int,
        reference_stable: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        F_n = self.F_func(features, U_seg)
        N_n = np.zeros((n_points, ne))
        if reference_stable:
            return F_n, N_n
        N_n_list = self.N_func(features, U_seg)
        for eq_idx, n_val in enumerate(N_n_list):
            N_n[:, eq_idx] = np.array(n_val).flatten()[:n_points]
        return F_n, N_n

    def time_step(
        self,
        U_n: np.ndarray,
        U_seg: List[np.ndarray],
        dt: float,
        coeffs_n: np.ndarray = None,
        **kwargs,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute one pseudo-time step using implicit Euler."""
        _ = (U_n, coeffs_n)
        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        self.U_seg_current = U_seg
        coeffs = self.fit(U_n_seg=U_seg, dt=dt, **kwargs)
        U_new, U_seg_new = self.construct(self.data, self._current_model, coeffs)
        self.U_seg_prev = self.U_seg_current
        self.U_seg_current = U_seg_new
        return U_new, U_seg_new, coeffs
