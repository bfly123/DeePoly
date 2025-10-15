import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
from src.abstract_class.base_net import BaseNet
from src.abstract_class.boundary_conditions import BoundaryConditionMixin


class LinearPDENet(BoundaryConditionMixin, BaseNet):
    """Linear partial differential equation solving network"""

    # prepare_gpu_data is now unified in BaseNet - no need to override

    def physics_loss(self, data_GPU: Dict) -> torch.Tensor:
        """Calculate physics loss function for linear PDE

        For linear PDEs: L1(u) - S = 0, where S is the precomputed source term

        Args:
            data_GPU: Data dictionary on GPU

        Returns:
            loss: Loss tensor
        """
        # Get training coordinates, source term, and model predictions
        x_train = data_GPU.get("x_train", None)
        source = data_GPU.get("source", None)
        _, U = self(x_train)

        # Compute all PDE operators using unified method from BaseNet
        ops = self.compute_pde_operators(U, x_train)
        L1, S = ops['L1'], ops['S']

        # Linear PDE residual: L1(u) - source = 0
        pde_residual = L1[0] - source[:,0]
        pde_loss = torch.mean(pde_residual**2)

        # Add boundary conditions using unified method
        return self.compute_total_loss(pde_loss, data_GPU, boundary_weight=10.0)
