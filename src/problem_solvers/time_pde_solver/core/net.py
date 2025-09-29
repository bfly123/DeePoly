import torch
import numpy as np
from typing import Dict
from src.abstract_class.base_net import BaseNet
from src.abstract_class.constants import pi
from src.abstract_class.boundary_conditions import BoundaryConditionMixin

class TimePDENet(BoundaryConditionMixin, BaseNet):
    """Neural network implementation for time-dependent problems"""
    
    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """Compute physics loss
        
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
        step = kwargs.get('step', 'pre')
        
        # Get training data
        x_train = data_GPU["x_train"]
        x_bd = data_GPU["x_bd"]
        u_bd = data_GPU["u_bd"]
        param = data_GPU["param"]
        U_n = data_GPU["U_current"]

        # Get model predictions
        _, U = self(x_train)

        # Get parameters
        Re = param[0]["Re"]
        nu = param[0]["nu"]

# auto code begin
# Config signature: {"F": [], "L1": ["-0.0001*diff(u,x,2)"], "L2": [], "N": ["-5*u+5*u**3"]}
        # Extract physical quantities from output
        u = U[..., 0]

        # Calculate derivatives in each direction
        du_x = self.gradients(u, x_train)[0][..., 0]

        # Calculate 2nd-order derivatives
        du_xx = self.gradients(du_x, x_train)[0][..., 0]

        # L1 operators
        L1 = [-0.0001*du_xx]

        # L2 operators
        L2 = [
        ]

        # F operators
        F = [
        ]

        # N operators
        N = [-5*u+5*u**3]

# auto code end

        # Using first-order forward Euler scheme: u^{n+1} = u^n + dt * [L1(u^{n+1}) + L2(u^n)*F(u^n)]
        # For neural network training, we directly minimize PDE residual
        # PDE residual: du/dt - L1(u) - L2(u)*F(u) = 0
        # Using first-order difference approximation: (u^{n+1} - u^n)/dt - L1(u^{n+1}) - L2(u^n)*F(u^n) = 0
        
        
        pde_loss = 0.0
        #dt = 0.01
        
        # Unified time evolution processing - eliminate index checking branches
        # Pad operators to match n_eqs length for consistent processing
        L1_padded = L1 + [0.0] * (self.config.n_eqs - len(L1))
        L2_padded = L2 + [0.0] * (self.config.n_eqs - len(L2))
        F_padded = F + [0.0] * (self.config.n_eqs - len(F))
        N_padded = N + [0.0] * (self.config.n_eqs - len(N))

        for i in range(self.config.n_eqs):
            # Unified operator application - no conditional branches
            l1_term = L1_padded[i]
            l2f_term = L2_padded[i] * F_padded[i]
            n_term = N_padded[i]

            # PDE residual computation
            residual_i = (U[:,i] - U_n[:,i]) + dt * (l1_term + l2f_term + n_term)
            pde_loss += torch.mean(residual_i**2)


        # Boundary conditions handling
        boundary_loss = self._compute_boundary_loss(data_GPU)
        boundary_loss_weight = 10.0
        
        # Total loss function
        total_loss = pde_loss + boundary_loss_weight * boundary_loss
        return total_loss
    
    # Boundary condition processing is now handled by BoundaryConditionMixin
    
    def prepare_gpu_data(self, data: Dict, U_current: np.ndarray = None) -> Dict:
        """Prepare GPU data for time PDE problems
        
        Args:
            data: Input data dictionary containing training data
            
        Returns:
            gpu_data: Dictionary containing GPU tensors
        """
        gpu_data = {}
        
        # Transfer coordinate data to GPU
        gpu_data["x_train"] = torch.tensor(
            data["x"], dtype=torch.float64, device=self.config.device, requires_grad=True
        )
        gpu_data["U_current"] = torch.tensor(
            U_current, dtype=torch.float64, device=self.config.device
        )
        
        # Use unified boundary condition processing
        boundary_gpu_data = self.prepare_boundary_gpu_data(data)
        gpu_data.update(boundary_gpu_data)
        
        # Transfer boundary points and values for physics loss computation
        if "global_boundary_dict" in data and data["global_boundary_dict"]:
            # Extract boundary data for easier access in physics_loss - Pure abstract U processing
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
        gpu_data["param"] = [{"Re": 1.0, "nu": 1.0}]  # Default parameters for time PDE
        
        return gpu_data
    
    @staticmethod
    def model_init(config):
        """Initialize model
        
        Args:
            config: Configuration object
            
        Returns:
            TimePDENet: Initialized model
        """
        model = TimePDENet(
            in_dim=config.n_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.n_eqs
        ).to(config.device)
        return model 