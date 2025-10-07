import torch
import numpy as np
from typing import Dict
from src.abstract_class.base_net import BaseNet
from src.abstract_class.constants import pi
from src.abstract_class.boundary_conditions import BoundaryConditionMixin

class GeneralPDENet(BoundaryConditionMixin, BaseNet):
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
        step = kwargs.get('step', 'pre')

        # Get training data
        x_train = data_GPU["x_train"]
        x_bd = data_GPU["x_bd"]
        u_bd = data_GPU["u_bd"]
        param = data_GPU["param"]
        U_n = data_GPU.get("U_current", None)

        # Get model predictions
        _, U = self(x_train)

        # Get parameters
        Re = param[0]["Re"]
        nu = param[0]["nu"]

# auto code begin
# Config signature: {"F": ["0"], "L1": ["diff(u,t)-0.0001*diff(u,x,2)"], "L2": ["0"], "N": ["-5*u+5*u**3"]}
        # Extract physical quantities from output
        u = U[..., 0]

        # Calculate 1st-order derivatives (对所有空间维度)
        # x_train shape: [N, 2] where dim 0=x, dim 1=t
        du_x = self.gradients(u, x_train)[0][..., 0]  # ∂u/∂x
        du_t = self.gradients(u, x_train)[0][..., 1]  # ∂u/∂t (t是空间维度)

        # Calculate 2nd-order derivatives
        du_xx = self.gradients(du_x, x_train)[0][..., 0]  # ∂²u/∂x²

        # L1 operators: diff(u,t) - 0.0001*diff(u,x,2)
        L1 = [du_t - 0.0001*du_xx]

        # L2 operators
        L2 = [0]

        # F operators
        F = [0]

        # N operators
        N = [-5*u+5*u**3]

        # S operators (source terms)
        S = [0]

# auto code end

        # Pseudo-time stepping formulation (参考 time_pde_solver):
        # ∂U/∂τ + R(U) = 0
        # Discretization: (U^{n+1} - U^n) / dτ + R(U^{n+1}) = 0
        # Rearranged: (U^{n+1} - U^n) + dτ · R(U^{n+1}) = 0
        #
        # During Spotter stage (dt=0 or U_n=None), directly minimize PDE residual R(U)=0
        # During Fitter stage (dt>0 and U_n provided), minimize pseudo-time residual

        pde_loss = 0.0

        # Directly use operators (config should define correct number of equations)
        for i in range(self.config.n_eqs):
            # Compute PDE residual: R(U) = L1 + L2*F + N - S
            l1_term = L1[i]
            l2f_term = L2[i] * F[i]
            n_term = N[i]
            s_term = S[i]

            # PDE residual R(U)
            pde_residual = l1_term + l2f_term + n_term - s_term

            # Add pseudo-time stepping term (类似 time_pde_solver)
            if dt > 0 and U_n is not None:
                # Pseudo-time residual: (U - U_n) + dt * R(U) = 0
                residual_i = (U[:,i] - U_n[:,i]) + dt * pde_residual
            else:
                # Spotter stage: directly minimize PDE residual R(U) = 0
                residual_i = pde_residual

            pde_loss += torch.mean(residual_i**2)
        boundary_loss_weight = 10
        # Boundary conditions handling
        boundary_loss = self._compute_boundary_loss(data_GPU)
        boundary_loss_weight = 10.0

        # Total loss function
        total_loss = pde_loss + boundary_loss_weight * boundary_loss
        return total_loss

    def prepare_gpu_data(self, data: Dict, U_current: np.ndarray = None) -> Dict:
        """Prepare GPU data for general PDE problems with pseudo-time stepping

        Args:
            data: Input data dictionary containing training data
            U_current: Current solution at pseudo-time τ^n (for pseudo-time stepping)

        Returns:
            gpu_data: Dictionary containing GPU tensors
        """
        gpu_data = {}

        # Transfer current solution for pseudo-time stepping (类似 time_pde_solver)
        if U_current is not None:
            gpu_data["U_current"] = torch.tensor(
                U_current, dtype=torch.float64, device=self.config.device
            )
        else:
            # Spotter stage: initialize with zeros
            n_points = len(data["x"])
            gpu_data["U_current"] = torch.zeros(
                (n_points, self.config.n_eqs), dtype=torch.float64, device=self.config.device
            )

        # Transfer coordinate data to GPU
        gpu_data["x_train"] = torch.tensor(
            data["x"], dtype=torch.float64, device=self.config.device, requires_grad=True
        )

        # For Spotter initialization, U_current is not needed
        if U_current is None:
            U_current = np.zeros((len(data["x"]), self.config.n_eqs))

        gpu_data["U_current"] = torch.tensor(
            U_current, dtype=torch.float64, device=self.config.device
        )

        # Use unified boundary condition processing
        boundary_gpu_data = self.prepare_boundary_gpu_data(data)
        gpu_data.update(boundary_gpu_data)

        # Transfer boundary points and values for physics loss computation
        if "global_boundary_dict" in data and data["global_boundary_dict"]:
            # Extract boundary data for easier access in physics_loss
            x_bd_list = []
            u_bd_list = []
            for var_idx in data["global_boundary_dict"]:
                for bc_type in data["global_boundary_dict"][var_idx]:
                    if "x" in data["global_boundary_dict"][var_idx][bc_type]:
                        x_bd = data["global_boundary_dict"][var_idx][bc_type]["x"]
                        u_bd = data["global_boundary_dict"][var_idx][bc_type]["values"]
                        if isinstance(x_bd, np.ndarray) and x_bd.size > 0:
                            x_bd_list.append(x_bd)
                            u_bd_list.append(u_bd)

            if x_bd_list:
                gpu_data["x_bd"] = torch.tensor(
                    np.vstack(x_bd_list), dtype=torch.float64, device=self.config.device, requires_grad=True
                )
                gpu_data["u_bd"] = torch.tensor(
                    np.vstack(u_bd_list), dtype=torch.float64, device=self.config.device
                )
            else:
                # Create empty tensors if no boundary data
                gpu_data["x_bd"] = torch.zeros((0, self.config.n_dim), dtype=torch.float64, device=self.config.device)
                gpu_data["u_bd"] = torch.zeros((0, self.config.n_eqs), dtype=torch.float64, device=self.config.device)
        else:
            # Create empty tensors if no boundary data
            gpu_data["x_bd"] = torch.zeros((0, self.config.n_dim), dtype=torch.float64, device=self.config.device)
            gpu_data["u_bd"] = torch.zeros((0, self.config.n_eqs), dtype=torch.float64, device=self.config.device)

        # Add parameter data for physics loss
        gpu_data["param"] = [{"Re": 1.0, "nu": 1.0}]  # Default parameters

        return gpu_data

    @staticmethod
    def model_init(config):
        """Initialize model

        Args:
            config: Configuration object

        Returns:
            GeneralPDENet: Initialized model
        """
        model = GeneralPDENet(
            in_dim=config.n_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.n_eqs
        ).to(config.device)
        return model