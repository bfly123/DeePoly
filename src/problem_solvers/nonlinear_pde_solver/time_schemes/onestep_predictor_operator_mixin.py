class OneStepPredictorOperatorMixin:
    def _predictor_step(self, U_n_seg, dt: float, step: int):
        """Run the implicit predictor solve and cache operator values."""
        predictor_data = {
            "U_n_seg": U_n_seg,
            "dt": dt,
            "step": step,
            "operation": "onestep_predictor",
        }
        coeffs_star = self.fitter.fit(**predictor_data)
        _, U_star_seg = self.fitter.construct(
            self.fitter.data,
            self.fitter._current_model,
            coeffs_star,
        )
        self.L1_beta_star, self.L2_beta_star = self._compute_operator_values(coeffs_star)
        self.predictor_solution = U_star_seg
        self.predictor_coeffs = coeffs_star
        return U_star_seg, coeffs_star

    def _compute_operator_values(self, coeffs):
        """Compute and cache L1*beta and L2*beta for every segment/equation."""
        L1_beta = []
        L2_beta = []
        for seg_idx in range(self.fitter.ns):
            L1_ops = self.fitter._linear_operators[seg_idx].get("L1")
            L2_ops = self.fitter._linear_operators[seg_idx].get("L2")
            L1_seg = []
            L2_seg = []
            for eq_idx in range(self.config.n_eqs):
                beta = coeffs[seg_idx, eq_idx, :]
                L1_seg.append(L1_ops[eq_idx] @ beta)
                L2_seg.append(L2_ops[eq_idx] @ beta)
            L1_beta.append(L1_seg)
            L2_beta.append(L2_seg)
        return L1_beta, L2_beta
