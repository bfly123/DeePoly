from typing import List

import numpy as np


class PseudoTimeFitterFormulationMixin:
    def _get_pseudo_time_direction(self) -> float:
        return float(getattr(self.config, "pseudo_time_direction", -1.0))

    def _get_residual_coefficient(self) -> float:
        direction = self._get_pseudo_time_direction()
        if direction == 0.0:
            raise ValueError("pseudo_time_direction must be non-zero")
        return -direction

    def _build_equation_jacobian(
        self,
        features_list: List[np.ndarray],
        U_seg: np.ndarray,
        features: np.ndarray,
        dt: float,
        L1: np.ndarray,
        L2: np.ndarray,
        F_eq: np.ndarray,
        steady_state_mode: bool,
        reference_stable: bool,
        eq_idx: int,
        ne: int,
        dgN: int,
    ) -> np.ndarray:
        residual_coeff = self._get_residual_coefficient()
        linear_block = L1 + np.diag(F_eq) @ L2
        if reference_stable:
            jacobian = residual_coeff * dt * linear_block
            jacobian += residual_coeff * dt * self._build_frozen_nonlinear_block(
                eq_idx=eq_idx,
                features_list=features_list,
                U_seg=U_seg,
                ne=ne,
                dgN=dgN,
            )
            if self._equation_uses_mass_term(eq_idx, steady_state_mode):
                jacobian += self._build_mass_block(
                    features,
                    self._get_mass_var_index(eq_idx),
                    ne,
                    dgN,
                )
            return jacobian

        if steady_state_mode:
            return linear_block.copy()

        return (
            self._build_mass_block(
                features,
                self._get_mass_var_index(eq_idx),
                ne,
                dgN,
            )
            / dt
            + residual_coeff * linear_block
            + residual_coeff
            * self._build_nonlinear_linearization_block(
                eq_idx=eq_idx,
                features=features,
                U_seg=U_seg,
                ne=ne,
                dgN=dgN,
            )
        )

    @staticmethod
    def _build_mass_block(
        features: np.ndarray,
        var_idx: int,
        ne: int,
        dgN: int,
    ) -> np.ndarray:
        mass_block = np.zeros((features.shape[0], ne * dgN))
        start = var_idx * dgN
        mass_block[:, start : start + dgN] = features
        return mass_block

    def _build_equation_rhs(
        self,
        U_seg: np.ndarray,
        dt: float,
        source_eq: np.ndarray,
        nonlinear_eq: np.ndarray,
        steady_state_mode: bool,
        reference_stable: bool,
        eq_idx: int,
    ) -> np.ndarray:
        residual_coeff = self._get_residual_coefficient()
        if reference_stable:
            rhs = residual_coeff * dt * source_eq
            if self._equation_uses_mass_term(eq_idx, steady_state_mode):
                rhs = rhs + U_seg[:, self._get_mass_var_index(eq_idx)]
            return rhs

        if steady_state_mode:
            return source_eq - nonlinear_eq

        return (
            U_seg[:, self._get_mass_var_index(eq_idx)] / dt
            + residual_coeff
            * (
                source_eq
                - nonlinear_eq
                + self._build_nonlinear_linearization_rhs_correction(
                    eq_idx=eq_idx,
                    U_seg=U_seg,
                )
            )
        )

    def _uses_reference_stable_formulation(self) -> bool:
        return getattr(self.config, "pseudo_time_formulation", "default") == "reference_stable"

    def _equation_uses_mass_term(self, eq_idx: int, steady_state_mode: bool) -> bool:
        if self._uses_reference_stable_formulation():
            mass_mask = {int(idx) for idx in getattr(self.config, "mass_eq_mask", [])}
            if mass_mask:
                return eq_idx in mass_mask
        return not steady_state_mode

    def _get_mass_var_index(self, eq_idx: int) -> int:
        mass_var_map = getattr(self.config, "mass_var_map", [])
        if eq_idx < len(mass_var_map) and int(mass_var_map[eq_idx]) >= 0:
            return int(mass_var_map[eq_idx])

        frozen_specs = self._get_frozen_nonlinear_specs()
        target_vars = {spec[2] for spec in frozen_specs.get(eq_idx, [])}
        if len(target_vars) == 1:
            return next(iter(target_vars))
        return eq_idx
