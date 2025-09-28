import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
from src.abstract_class.base_net import BaseNet


class LinearPDENet(BaseNet):
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

        # Transfer global_boundary_dict data to GPU - 纯AbstractUProcess
        global_boundary_dict = {}
        for var_idx in data["global_boundary_dict"]:
            global_boundary_dict[var_idx] = {}
            for bc_type in data["global_boundary_dict"][var_idx]:
                global_boundary_dict[var_idx][bc_type] = {}
                for key, value in data["global_boundary_dict"][var_idx][bc_type].items():
                    if isinstance(value, np.ndarray) and value.size > 0:
                        # Boundary conditionspoint可能也NeedGradient，Especially用于Neumann或RobinCondition
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

        # Initialize boundary loss
        boundary_loss = 0.0
        boundary_loss_weight = 10.0  # Weight for boundary conditions

        # Process boundary conditions if they exist - 纯AbstractUProcess
        if global_boundary_dict:
            # Process each U component in the boundary conditions
            for var_idx in global_boundary_dict:
                # Process Dirichlet boundary conditions
                if (
                    "dirichlet" in global_boundary_dict[var_idx]
                    and global_boundary_dict[var_idx]["dirichlet"]["x"].shape[0] > 0
                ):
                    x_bc = global_boundary_dict[var_idx]["dirichlet"]["x"]
                    u_bc = global_boundary_dict[var_idx]["dirichlet"]["values"]

                    # Get model predictions at boundary points
                    _, pred_bc = self(x_bc)

                    # Calculate boundary loss (MSE between predictions and target values)
                    bc_error = (pred_bc - u_bc) ** 2
                    boundary_loss += torch.mean(bc_error)

                # Process Neumann boundary conditions
                if (
                    "neumann" in global_boundary_dict[var_idx]
                    and global_boundary_dict[var_idx]["neumann"]["x"].shape[0] > 0
                ):
                    x_bc = global_boundary_dict[var_idx]["neumann"]["x"]
                    u_bc = global_boundary_dict[var_idx]["neumann"]["values"]
                    normals = global_boundary_dict[var_idx]["neumann"]["normals"]

                    # Store all normal derivative errors for all boundary points
                    all_derivatives_errors = []

                    # Calculate normal derivative at each boundary point
                    for i in range(x_bc.shape[0]):
                        x_point = x_bc[i : i + 1].clone().detach().requires_grad_(True)

                        # Get prediction
                        _, u_pred = self(x_point)

                        # Calculate gradient
                        grads = self.gradients(u_pred, x_point)[0]

                        # Calculate normal derivative (dot product of gradient and normal vector)
                        normal = normals[i]
                        normal_derivative = torch.sum(grads * normal)

                        # Calculate error
                        bc_error = (normal_derivative - u_bc[i]) ** 2
                        all_derivatives_errors.append(bc_error)

                    # Calculate MSE for all Neumann boundary points
                    if all_derivatives_errors:
                        neumann_errors = torch.stack(all_derivatives_errors)
                        boundary_loss += torch.mean(neumann_errors)

                # Process Robin boundary conditions
                if (
                    "robin" in global_boundary_dict[var_idx]
                    and global_boundary_dict[var_idx]["robin"]["x"].shape[0] > 0
                ):
                    x_bc = global_boundary_dict[var_idx]["robin"]["x"]
                    u_bc = global_boundary_dict[var_idx]["robin"]["values"]
                    normals = global_boundary_dict[var_idx]["robin"]["normals"]
                    params = global_boundary_dict[var_idx]["robin"]["params"]

                    # Store all Robin boundary condition errors
                    all_robin_errors = []

                    # Process each Robin boundary point
                    for i in range(x_bc.shape[0]):
                        x_point = x_bc[i : i + 1].clone().detach().requires_grad_(True)

                        # Get prediction
                        _, u_pred = self(x_point)

                        # Get parameters
                        alpha, beta = params[0], params[1]

                        # Calculate gradient if beta is non-zero
                        if abs(beta) > 1e-10:
                            grads = self.gradients(u_pred, x_point)[0]

                            # Calculate normal derivative
                            normal = normals[i]
                            normal_derivative = torch.sum(grads * normal)

                            # Robin condition: alpha*u + beta*du/dn = g
                            robin_value = alpha * u_pred + beta * normal_derivative
                            bc_error = (robin_value - u_bc[i]) ** 2
                        else:
                            # If beta is zero, it's effectively a Dirichlet condition
                            bc_error = (alpha * u_pred - u_bc[i]) ** 2
                        
                        all_robin_errors.append(bc_error)
                    
                    # Calculate MSE for all Robin boundary points
                    if all_robin_errors:
                        robin_errors = torch.stack(all_robin_errors)
                        boundary_loss += torch.mean(robin_errors)
                
                # Process periodic boundary conditions
                if 'periodic' in global_boundary_dict[var_idx] and global_boundary_dict[var_idx]['periodic']['pairs']:
                    for pair in global_boundary_dict[var_idx]['periodic']['pairs']:
                        x_bc_1 = pair['x_1']
                        x_bc_2 = pair['x_2']
                        constraint_type = pair['constraint_type']
                        
                        # Get model predictions at both boundary regions
                        _, pred_bc_1 = self(x_bc_1)
                        _, pred_bc_2 = self(x_bc_2)
                        
                        if constraint_type == 'dirichlet':
                            # Periodic Dirichlet: U(x1) = U(x2)
                            periodic_error = (pred_bc_1 - pred_bc_2) ** 2
                            boundary_loss += torch.mean(periodic_error)
                        elif constraint_type == 'neumann':
                            # Periodic Neumann: ∂U/∂n(x1) = ∂U/∂n(x2)
                            normals_1 = pair['normals_1']
                            normals_2 = pair['normals_2']
                            
                            # Calculate normal derivatives at both boundary regions
                            all_periodic_errors = []
                            for i in range(x_bc_1.shape[0]):
                                x_point_1 = x_bc_1[i:i+1].clone().detach().requires_grad_(True)
                                x_point_2 = x_bc_2[i:i+1].clone().detach().requires_grad_(True)
                                
                                _, u_pred_1 = self(x_point_1)
                                _, u_pred_2 = self(x_point_2)
                                
                                grads_1 = self.gradients(u_pred_1, x_point_1)[0]
                                grads_2 = self.gradients(u_pred_2, x_point_2)[0]
                                
                                normal_1 = normals_1[i]
                                normal_2 = normals_2[i]
                                
                                normal_deriv_1 = torch.sum(grads_1 * normal_1)
                                normal_deriv_2 = torch.sum(grads_2 * normal_2)
                                
                                periodic_error = (normal_deriv_1 - normal_deriv_2) ** 2
                                all_periodic_errors.append(periodic_error)
                            
                            if all_periodic_errors:
                                periodic_errors = torch.stack(all_periodic_errors)
                                boundary_loss += torch.mean(periodic_errors)

        # Combine losses with appropriate weights
        total_loss = pde_loss + boundary_loss_weight * boundary_loss


        return total_loss
