import torch
import numpy as np
from typing import Dict
from src.abstract_class.base_net import BaseNet
from src.abstract_class.constants import pi

class TimePDENet(BaseNet):
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
    
    def _compute_boundary_loss(self, data_GPU: Dict) -> torch.Tensor:
        """Unified boundary condition loss computation - eliminate type branching"""
        global_boundary_dict = data_GPU.get("global_boundary_dict", {})

        # Boundary condition type handlers - unified interface
        bc_handlers = {
            "dirichlet": self._compute_dirichlet_loss,
            "neumann": self._compute_neumann_loss,
            "robin": self._compute_robin_loss,
            "periodic": self._compute_periodic_loss,
        }

        total_boundary_loss = 0.0
        for var_idx, var_bc_dict in global_boundary_dict.items():
            for bc_type, handler in bc_handlers.items():
                if bc_type in var_bc_dict and self._has_valid_bc_data(var_bc_dict[bc_type]):
                    total_boundary_loss += handler(var_bc_dict[bc_type])

        return total_boundary_loss
    
    def _has_valid_bc_data(self, bc_data: Dict) -> bool:
        """Unified boundary condition data validation - eliminate type-specific checks"""
        if not bc_data:
            return False

        # Check for standard data structure
        if "x" in bc_data:
            return bc_data["x"].shape[0] > 0

        # Check for periodic-specific structure
        if "pairs" in bc_data:
            return bool(bc_data["pairs"])

        return False
    
    def _compute_dirichlet_loss(self, bc_data: Dict) -> torch.Tensor:
        """Compute Dirichlet boundary condition loss"""
        x_bc = bc_data["x"]
        u_bc = bc_data["values"]
        
        _, pred_bc = self(x_bc)
        return torch.mean((pred_bc - u_bc) ** 2)
    
    def _compute_neumann_loss(self, bc_data: Dict) -> torch.Tensor:
        """Compute Neumann boundary condition loss"""
        x_bc = bc_data["x"]
        u_bc = bc_data["values"]
        normals = bc_data["normals"]
        
        errors = []
        for i in range(x_bc.shape[0]):
            x_point = x_bc[i:i+1].clone().detach().requires_grad_(True)
            _, u_pred = self(x_point)
            
            grads = self.gradients(u_pred, x_point)[0]
            normal_derivative = torch.sum(grads * normals[i])
            
            errors.append((normal_derivative - u_bc[i]) ** 2)
        
        return torch.mean(torch.stack(errors)) if errors else 0.0
    
    def _compute_robin_loss(self, bc_data: Dict) -> torch.Tensor:
        """Compute Robin boundary condition loss"""
        x_bc = bc_data["x"]
        u_bc = bc_data["values"]
        normals = bc_data["normals"]
        params = bc_data["params"]
        alpha, beta = params[0], params[1]
        
        errors = []
        for i in range(x_bc.shape[0]):
            x_point = x_bc[i:i+1].clone().detach().requires_grad_(True)
            _, u_pred = self(x_point)
            
            if abs(beta) > 1e-10:
                grads = self.gradients(u_pred, x_point)[0]
                normal_derivative = torch.sum(grads * normals[i])
                robin_value = alpha * u_pred + beta * normal_derivative
            else:
                robin_value = alpha * u_pred
                
            errors.append((robin_value - u_bc[i]) ** 2)
        
        return torch.mean(torch.stack(errors)) if errors else 0.0
    
    def _compute_periodic_loss(self, periodic_data: Dict) -> torch.Tensor:
        """统一的周期边界条件损失计算"""
        total_loss = 0.0

        for pair in periodic_data['pairs']:
            x_bc_1, x_bc_2 = pair['x_1'], pair['x_2']

            _, pred_bc_1 = self(x_bc_1)
            _, pred_bc_2 = self(x_bc_2)

            # 统一的周期边界条件: U(x1) = U(x2)
            total_loss += torch.mean((pred_bc_1 - pred_bc_2) ** 2)

        return total_loss
    
    # 移除_compute_periodic_neumann_loss方法 - 不再需要复杂的导数约束
    
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
        
        # Transfer boundary data to GPU - Pure abstract U processing
        if "global_boundary_dict" in data:
            global_boundary_dict = {}
            for var_idx in data["global_boundary_dict"]:
                global_boundary_dict[var_idx] = {}
                for bc_type in data["global_boundary_dict"][var_idx]:
                    global_boundary_dict[var_idx][bc_type] = {}
                    
                    if bc_type == 'periodic':
                        # Handle periodic boundary condition pairs
                        global_boundary_dict[var_idx][bc_type]['pairs'] = []
                        for pair in data["global_boundary_dict"][var_idx][bc_type]['pairs']:
                            gpu_pair = {}
                            for pair_key, pair_value in pair.items():
                                if isinstance(pair_value, np.ndarray) and pair_value.size > 0:
                                    if 'x_' in pair_key:  # x_1, x_2
                                        gpu_pair[pair_key] = torch.tensor(
                                            pair_value, dtype=torch.float64, device=self.config.device, requires_grad=True
                                        )
                                    else:  # normals_1, normals_2, etc.
                                        gpu_pair[pair_key] = torch.tensor(
                                            pair_value, dtype=torch.float64, device=self.config.device
                                        )
                                else:
                                    gpu_pair[pair_key] = pair_value
                            global_boundary_dict[var_idx][bc_type]['pairs'].append(gpu_pair)
                    else:
                        # Handle regular boundary conditions
                        for key, value in data["global_boundary_dict"][var_idx][bc_type].items():
                            if isinstance(value, np.ndarray) and value.size > 0:
                                if key == "x":
                                    global_boundary_dict[var_idx][bc_type][key] = torch.tensor(
                                        value, dtype=torch.float64, device=self.config.device, requires_grad=True
                                    )
                                else:
                                    global_boundary_dict[var_idx][bc_type][key] = torch.tensor(
                                        value, dtype=torch.float64, device=self.config.device
                                    )
                            else:
                                global_boundary_dict[var_idx][bc_type][key] = value
            gpu_data["global_boundary_dict"] = global_boundary_dict
        else:
            gpu_data["global_boundary_dict"] = {}
        
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