import numpy as np


class OneStepPredictorCorrectorMixin:
    def _corrector_step(
        self,
        U_n_seg,
        U_star_seg,
        dt: float,
        coeffs_star,
        step: int,
    ):
        """Apply the explicit corrector using precomputed operator values."""
        _ = step
        U_new_seg = []
        for seg_idx, (U_n, U_star) in enumerate(zip(U_n_seg, U_star_seg)):
            U_mid = 0.5 * (U_n + U_star)
            features = self.fitter._features[seg_idx][0]
            F_mid = self.fitter.F_func(features, U_mid)
            N_mid = self.fitter.N_func(features, U_mid)
            U_corrected = U_n.copy()
            for eq_idx in range(self.config.n_eqs):
                L1_avg = self._get_operator_average(self.L1_beta_star, self.L1_beta_prev, seg_idx, eq_idx, U_n.shape[0])
                L2_avg = self._get_operator_average(self.L2_beta_star, self.L2_beta_prev, seg_idx, eq_idx, U_n.shape[0])
                L2F_avg = L2_avg * F_mid[:, eq_idx]
                N_contrib = N_mid[eq_idx]
                U_corrected[:, eq_idx] -= dt * (L1_avg + L2F_avg + N_contrib)
            U_new_seg.append(U_corrected)
        return U_new_seg, coeffs_star

    @staticmethod
    def _get_operator_average(current, previous, seg_idx: int, eq_idx: int, n_points: int):
        if current is None or previous is None:
            return np.zeros(n_points)
        if len(current) <= seg_idx or len(previous) <= seg_idx:
            return np.zeros(n_points)
        if len(current[seg_idx]) <= eq_idx or len(previous[seg_idx]) <= eq_idx:
            return np.zeros(n_points)
        current_val = current[seg_idx][eq_idx]
        previous_val = previous[seg_idx][eq_idx]
        if current_val is None or previous_val is None:
            return np.zeros(n_points)
        return 0.5 * (current_val + previous_val)
