import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
from src.abstract_class.base_net import BaseNet
from src.abstract_class.boundary_conditions import BoundaryConditionMixin


class LinearPDENet(BoundaryConditionMixin, BaseNet):
    """Linear partial differential equation solving network"""

    def prepare_gpu_data(self, data: Dict) -> Dict:
        """Prepare GPU data

        Args:
            data: Input data dictionary

        Returns:
            gpu_data: Dictionary containing GPU tensors
        """
        gpu_data = {}

        gpu_data["x"] = torch.tensor(
            data["x"], dtype=torch.float64, device=self.config.device, requires_grad=True
        )
        gpu_data["source"] = torch.tensor(
            data["source"], dtype=torch.float64, device=self.config.device
        )

        # Use unified boundary condition processing
        boundary_gpu_data = self.prepare_boundary_gpu_data(data)
        gpu_data.update(boundary_gpu_data)

        return gpu_data

    def physics_loss(self, data_GPU: Dict) -> torch.Tensor:
        """Calculate physics loss function

        Args:
            data_GPU: Data dictionary on GPU

        Returns:
            loss: Loss tensor
        """
        x_train = data_GPU.get("x", None)
        source = data_GPU.get("source", None)
        global_boundary_dict = data_GPU.get("global_boundary_dict", None)


        _, output = self(x_train)
# auto code begin
# Config signature: {"F": ["0"], "L1": ["diff(u,x,2) + diff(u,y,2)"], "L2": ["0"], "N": ["0"]}
        # Extract physical quantities from output
        u = output[..., 0]

        # Calculate derivatives in each direction
        du_x = self.gradients(u, x_train)[0][..., 0]
        du_y = self.gradients(u, x_train)[0][..., 1]

        # Calculate 2nd-order derivatives
        du_xx = self.gradients(du_x, x_train)[0][..., 0]
        du_yy = self.gradients(du_y, x_train)[0][..., 1]

        # L1 operators
        L1 = [du_xx + du_yy]

        # L2 operators (not used in linear PDEs)
        L2 = []

        # F operators (not used in linear PDEs)
        F = []

        # N operators (not used in linear PDEs)
        N = []

# auto code end

        # Calculate PDE residual loss
        # For linear PDEs: L1(u) - S = 0, where S is the precomputed source term
        pde_residual = L1[0] - source[:,0]
        pde_loss = torch.mean(pde_residual**2)

        # Use unified boundary condition processing
        boundary_loss = self._compute_boundary_loss(data_GPU)
        boundary_loss_weight = 10.0  # Weight for boundary conditions

        # Combine losses with appropriate weights
        total_loss = pde_loss + boundary_loss_weight * boundary_loss


        return total_loss
