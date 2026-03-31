import torch
import numpy as np
from typing import Dict
from src.abstract_class.base_net import BaseNet
from src.abstract_class.constants import pi
from src.abstract_class.boundary_conditions import BoundaryConditionMixin

class FixIterationPDENet(BoundaryConditionMixin, BaseNet):
    """Neural network implementation for fixed-point iteration (steady-state PDE)

    This network is specifically designed for steady-state PDEs solved using
    fixed-point iteration, where we directly minimize the PDE residual without
    time-stepping.
    """

    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """Compute physics loss for steady-state PDE with fixed-point iteration

        For fixed-point iteration, we directly minimize the PDE residual:
            R(U) = L1(U) + L2(U)·F(U) + N(U) - S = 0

        Args:
            data_GPU: GPU data dictionary containing training data
            **kwargs: Additional parameters (not used for fixed-point iteration)

        Returns:
            torch.Tensor: Loss value
        """
        # Get training coordinates and model predictions
        x_train = data_GPU["x_train"]
        _, U = self(x_train)

        # Compute all PDE operators using unified method from BaseNet
        ops = self.pad_operator_outputs(self.compute_pde_operators(U, x_train))
        L1, L2, F, N, S = ops['L1'], ops['L2'], ops['F'], ops['N'], ops['S']

        # Fixed-point iteration: directly minimize PDE residual R(U) = 0
        pde_loss = 0.0
        for i in range(self.config.n_eqs):
            # PDE residual: R(U) = L1 + L2*F + N - S
            pde_residual = L1[i] + L2[i] * F[i] + N[i] - S[i]
            pde_loss += torch.mean(pde_residual**2)

        # Add boundary conditions using unified method
        boundary_weight = float(getattr(self.config, "boundary_constraint_weight", 10.0))
        return self.compute_total_loss(pde_loss, data_GPU, boundary_weight=boundary_weight)

    # prepare_gpu_data is now unified in BaseNet - no need to override

    @staticmethod
    def model_init(config):
        """Initialize model

        Args:
            config: Configuration object

        Returns:
            FixIterationPDENet: Initialized model
        """
        model = FixIterationPDENet(config=config).to(config.device)
        return model
