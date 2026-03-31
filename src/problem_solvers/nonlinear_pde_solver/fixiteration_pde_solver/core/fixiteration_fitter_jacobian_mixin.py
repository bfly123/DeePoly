from typing import Dict, Tuple

import numpy as np


class FixIterationFitterJacobianMixin:
    def _build_segment_jacobian(
        self,
        segment_idx: int,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the fixed-point Jacobian for one segment."""
        U_seg, n_points, ne, dgN, features_list, L1_ops, L2_ops = self._load_segment_context(
            segment_idx,
            kwargs,
        )
        F_n, N_n = self._evaluate_fixed_operators(features_list, U_seg, n_points, ne)
        source_term = self._get_source_term(segment_idx)
        system_matrix = np.zeros((ne * n_points, ne * dgN))
        rhs_vectors = []

        for eq_idx in range(ne):
            row_start = eq_idx * n_points
            row_end = (eq_idx + 1) * n_points
            system_matrix[row_start:row_end, :] = self._build_equation_matrix(
                L1_ops[eq_idx],
                L2_ops[eq_idx],
                F_n[:, eq_idx],
            )
            rhs_vectors.append(
                self._build_equation_rhs(
                    source_term[:, eq_idx],
                    N_n[:, eq_idx],
                )
            )

        return system_matrix, np.concatenate(rhs_vectors)

    def _load_segment_context(
        self,
        segment_idx: int,
        kwargs: Dict,
    ) -> Tuple[np.ndarray, int, int, int, list, list, list]:
        U_current = kwargs.get("U_current")
        if U_current is None:
            raise ValueError("U_current must be provided in kwargs for fixed-point iteration")

        U_seg = self.global_to_segments(U_current)[segment_idx]
        n_points = len(self.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.dgN
        features_list = self._features[segment_idx]
        linear_ops = self._linear_operators[segment_idx]
        return (
            U_seg,
            n_points,
            ne,
            dgN,
            features_list,
            linear_ops.get("L1", None),
            linear_ops.get("L2", None),
        )

    def _evaluate_fixed_operators(
        self,
        features_list,
        U_seg: np.ndarray,
        n_points: int,
        ne: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        F_n = self.F_func(features_list, U_seg)
        N_n_list = self.N_func(features_list, U_seg)
        N_n = np.zeros((n_points, ne))
        for eq_idx, n_val in enumerate(N_n_list):
            N_n[:, eq_idx] = np.asarray(n_val).reshape(-1)[:n_points]
        return F_n, N_n

    @staticmethod
    def _build_equation_matrix(
        L1: np.ndarray,
        L2: np.ndarray,
        F_eq: np.ndarray,
    ) -> np.ndarray:
        return L1.copy() + np.diag(F_eq) @ L2

    @staticmethod
    def _build_equation_rhs(
        source_eq: np.ndarray,
        nonlinear_eq: np.ndarray,
    ) -> np.ndarray:
        return source_eq - nonlinear_eq
