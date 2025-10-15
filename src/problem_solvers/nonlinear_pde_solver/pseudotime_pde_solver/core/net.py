import torch
import numpy as np
from typing import Dict
from src.abstract_class.base_net import BaseNet
from src.abstract_class.constants import pi
from src.abstract_class.boundary_conditions import BoundaryConditionMixin

class PseudoTimePDENet(BoundaryConditionMixin, BaseNet):
    """Neural network implementation for general PDE problems (steady-state)"""

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
        # Get optional parameters
        dt = kwargs.get('dt', 0)
        U_n = data_GPU.get("U_current", None)

        # Get training coordinates and model predictions
        x_train = data_GPU["x_train"]
        _, U = self(x_train)

        # Compute all PDE operators using unified method from BaseNet
        ops = self.compute_pde_operators(U, x_train)
        L1, L2, F, N, S = ops['L1'], ops['L2'], ops['F'], ops['N'], ops['S']

        # Pseudo-time stepping: (U^{n+1} - U^n) + dτ · R(U^{n+1}) = 0
        # During Spotter stage (dt=0), directly minimize R(U)=0
        # During Fitter stage (dt>0), minimize pseudo-time residual
        pde_loss = 0.0
        for i in range(self.config.n_eqs):
            # PDE residual: R(U) = L1 + L2*F + N - S
            pde_residual = L1[i] + L2[i] * F[i] + N[i] - S[i]

            # Pseudo-time residual: (U - U_n) + dt * R(U)
            residual_i = (U[:,i] - U_n[:,i]) + dt * pde_residual
            pde_loss += torch.mean(residual_i**2)

        # Add boundary conditions using unified method
        return self.compute_total_loss(pde_loss, data_GPU, boundary_weight=10.0)

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