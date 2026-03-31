import torch
import numpy as np
from typing import Dict
from src.abstract_class.base_net import BaseNet
from src.abstract_class.constants import pi
from src.abstract_class.boundary_conditions import BoundaryConditionMixin

class PseudoTimePDENet(BoundaryConditionMixin, BaseNet):
    """Neural network implementation for general PDE problems (steady-state)"""

    def _get_pseudo_time_direction(self) -> float:
        return float(getattr(self.config, "pseudo_time_direction", -1.0))

    def _get_residual_coefficient(self) -> float:
        direction = self._get_pseudo_time_direction()
        if direction == 0.0:
            raise ValueError("pseudo_time_direction must be non-zero")
        return -direction

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
        return eq_idx

    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """Compute physics loss for steady-state PDE with pseudo-time stepping

        Args:
            data_GPU: GPU data dictionary containing training data
            **kwargs: Additional parameters
                - dt: Pseudo-time step size (dτ)
                - step: Training step type

        Returns:
            torch.Tensor: Loss value
        """
        dt = kwargs.get('dt', 0)
        U_n = data_GPU.get("U_current", None)
        steady_state_mode = getattr(self.config, "steady_state_mode", False)

        # Get training coordinates and model predictions
        x_train = data_GPU["x_train"]
        _, U = self(x_train)

        # Compute all PDE operators using unified method from BaseNet
        ops = self.pad_operator_outputs(self.compute_pde_operators(U, x_train))
        L1, L2, F, N, S = ops['L1'], ops['L2'], ops['F'], ops['N'], ops['S']
        reference_stable = self._uses_reference_stable_formulation()
        residual_coeff = self._get_residual_coefficient()

        pde_loss = 0.0
        for i in range(self.config.n_eqs):
            pde_residual = L1[i] + L2[i] * F[i] + N[i] - S[i]
            if U_n is None or dt == 0:
                residual_i = pde_residual
            elif reference_stable:
                residual_i = residual_coeff * dt * pde_residual
                if self._equation_uses_mass_term(i, steady_state_mode):
                    mass_var_idx = self._get_mass_var_index(i)
                    residual_i = (U[:, mass_var_idx] - U_n[:, mass_var_idx]) + residual_i
            elif steady_state_mode:
                residual_i = pde_residual
            else:
                residual_i = (U[:, i] - U_n[:, i]) + residual_coeff * dt * pde_residual
            pde_loss += torch.mean(residual_i**2)

        # Add boundary conditions using unified method
        boundary_weight = float(getattr(self.config, "boundary_constraint_weight", 10.0))
        return self.compute_total_loss(pde_loss, data_GPU, boundary_weight=boundary_weight)

    # prepare_gpu_data is now unified in BaseNet - no need to override
    # Note: The original had a bug with duplicate U_current handling (lines 62-72 and 79-85)
    # This is now fixed in the unified implementation

    @staticmethod
    def model_init(config):
        """Initialize model

        Args:
            config: Configuration object

        Returns:
            PseudoTimePDENet: Initialized model
        """
        model = PseudoTimePDENet(config=config).to(config.device)
        return model
