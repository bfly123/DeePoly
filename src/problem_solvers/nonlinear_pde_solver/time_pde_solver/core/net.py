import torch
import numpy as np
from typing import Dict
from src.abstract_class.base_net import BaseNet
from src.abstract_class.constants import pi
from src.abstract_class.boundary_conditions import BoundaryConditionMixin

class TimePDENet(BoundaryConditionMixin, BaseNet):
    """Neural network implementation for time-dependent problems"""
    
    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """Compute physics loss for time-dependent PDE

        Time-stepping formulation: (U^{n+1} - U^n)/dt = L1(U^{n+1}) + L2(U^n)*F(U^n) + N(U^{n+1})

        Args:
            data_GPU: GPU data dictionary containing training data
            **kwargs: Additional parameters
                - dt: Time step size, default 0
                - step: Time step type, default "pre"

        Returns:
            torch.Tensor: Loss value
        """
        # Get optional parameters
        dt = kwargs.get('dt', 0)
        U_n = data_GPU["U_current"]

        # Get training coordinates and model predictions
        x_train = data_GPU["x_train"]
        _, U = self(x_train)

        # Compute all PDE operators using unified method from BaseNet
        ops = self.compute_pde_operators(U, x_train)
        L1, L2, F, N, S = ops['L1'], ops['L2'], ops['F'], ops['N'], ops['S']

        # Time-stepping residual: (U^{n+1} - U^n) + dt * [L1 + L2*F + N - S] = 0
        pde_loss = 0.0
        for i in range(self.config.n_eqs):
            # PDE residual computation
            residual_i = (U[:,i] - U_n[:,i]) + dt * (L1[i] + L2[i] * F[i] + N[i] - S[i])
            pde_loss += torch.mean(residual_i**2)

        # Add boundary conditions using unified method
        return self.compute_total_loss(pde_loss, data_GPU, boundary_weight=10.0)

    # prepare_gpu_data is now unified in BaseNet - no need to override

    @staticmethod
    def model_init(config):
        """Initialize model

        Args:
            config: Configuration object

        Returns:
            TimePDENet: Initialized model
        """
        model = TimePDENet(config=config).to(config.device)
        return model 