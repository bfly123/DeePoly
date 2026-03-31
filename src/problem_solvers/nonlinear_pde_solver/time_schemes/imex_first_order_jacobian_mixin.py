from typing import Dict, Tuple

import numpy as np


class ImexFirstOrderJacobianMixin:
    def build_stage_jacobian(self, segment_idx: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.fitter.dgN
        dt = kwargs.get("dt")
        L1_operators = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)
        features_list = self.fitter._features[segment_idx]
        features = features_list[0]
        U_n_seg = kwargs.get("U_n_seg", [])[segment_idx]
        F_n = self.fitter.F_func(features_list, U_n_seg)
        N_n = self.fitter.N_func(features_list, U_n_seg)
        V = features_list[0]
        rhs = self._build_stage_rhs(
            segment_idx,
            L1_operators=L1_operators,
            L2_operators=L2_operators,
            U_n_seg=U_n_seg,
            F_n=F_n,
            features=features,
            N_n=N_n,
            stage=1,
            dt=dt,
        )
        return self._assemble_jacobian_system(
            ne=ne,
            n_points=n_points,
            dgN=dgN,
            V=V,
            dt=dt,
            L1_operators=L1_operators,
            L2_operators=L2_operators,
            F_n=F_n,
            rhs=rhs,
        )

    def _assemble_jacobian_system(
        self,
        *,
        ne: int,
        n_points: int,
        dgN: int,
        V: np.ndarray,
        dt: float,
        L1_operators,
        L2_operators,
        F_n,
        rhs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        L_final = np.zeros((ne * n_points, ne * dgN))
        b_final = []
        for eq_idx in range(ne):
            row_start = eq_idx * n_points
            row_end = (eq_idx + 1) * n_points
            col_start = eq_idx * dgN
            col_end = (eq_idx + 1) * dgN
            J_eq = self._build_equation_jacobian(
                V=V,
                dt=dt,
                L1=L1_operators[eq_idx],
                L2=L2_operators[eq_idx],
                F_eq=F_n[:, eq_idx],
            )
            L_final[row_start:row_end, col_start:col_end] = J_eq
            b_final.append(rhs[:, eq_idx])
        return L_final, np.concatenate(b_final)

    @staticmethod
    def _build_equation_jacobian(*, V: np.ndarray, dt: float, L1, L2, F_eq) -> np.ndarray:
        J_eq = V.copy()
        J_eq += dt * L1
        J_eq += dt * np.diag(F_eq) @ L2
        return J_eq

    def _build_stage_rhs(self, segment_idx: int, **kwargs) -> np.ndarray:
        del segment_idx
        U_n_seg = kwargs.get("U_n_seg")
        N_n = kwargs.get("N_n")
        dt = kwargs.get("dt")
        ne = self.config.n_eqs
        rhs = np.zeros((U_n_seg.shape[0], ne))
        rhs[:, :] = U_n_seg
        for eq_idx in range(ne):
            rhs[:, eq_idx] -= dt * N_n[eq_idx]
        return rhs
