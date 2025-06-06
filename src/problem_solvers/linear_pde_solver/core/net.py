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

        # Transfer global_boundary_dict data to GPU
        global_boundary_dict = {}
        for var in data["global_boundary_dict"]:
            global_boundary_dict[var] = {}
            for bc_type in data["global_boundary_dict"][var]:
                global_boundary_dict[var][bc_type] = {}
                for key, value in data["global_boundary_dict"][var][bc_type].items():
                    if isinstance(value, np.ndarray) and value.size > 0:
                        # 边界条件点可能也需要梯度，特别是用于Neumann或Robin条件
                        if key == "x":
                            global_boundary_dict[var][bc_type][key] = torch.tensor(
                                value, dtype=torch.float64, device=self.config.device, requires_grad=True
                            )
                        else:
                            global_boundary_dict[var][bc_type][key] = torch.tensor(
                                value, dtype=torch.float64, device=self.config.device
                            )
                    else:
                        global_boundary_dict[var][bc_type][key] = value

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
        # Extract physical quantities from output
        u = output[..., 0]

        # Calculate derivatives in each direction
        du_x = self.gradients(u, x_train)[0][..., 0]
        du_y = self.gradients(u, x_train)[0][..., 1]

        # Calculate second-order derivatives
        du_xx = self.gradients(du_x, x_train)[0][..., 0]
        du_yy = self.gradients(du_y, x_train)[0][..., 1]

        # Compute equations as a list
        eq = [du_xx + du_yy]

        pde_loss = torch.mean((eq[0] - source[0]) ** 2)

# auto code end

        pde_loss = 0.0
        for i in range(len(eq)):
            pde_loss += torch.mean((eq[i] - source[:,i]) ** 2)

        # Initialize boundary loss
        boundary_loss = 0.0
        boundary_loss_weight = 10.0  # Weight for boundary conditions

        # Process boundary conditions if they exist
        if global_boundary_dict:
            # Process each variable in the boundary conditions
            for var in global_boundary_dict:
                # Process Dirichlet boundary conditions
                if (
                    "dirichlet" in global_boundary_dict[var]
                    and global_boundary_dict[var]["dirichlet"]["x"].shape[0] > 0
                ):
                    x_bc = global_boundary_dict[var]["dirichlet"]["x"]
                    u_bc = global_boundary_dict[var]["dirichlet"]["u"]

                    # Get model predictions at boundary points
                    _, pred_bc = self(x_bc)

                    # Calculate boundary loss (MSE between predictions and target values)
                    bc_error = (pred_bc - u_bc) ** 2
                    boundary_loss += torch.mean(bc_error)

                # Process Neumann boundary conditions
                if (
                    "neumann" in global_boundary_dict[var]
                    and global_boundary_dict[var]["neumann"]["x"].shape[0] > 0
                ):
                    x_bc = global_boundary_dict[var]["neumann"]["x"]
                    u_bc = global_boundary_dict[var]["neumann"]["u"]
                    normals = global_boundary_dict[var]["neumann"]["normals"]

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
                    "robin" in global_boundary_dict[var]
                    and global_boundary_dict[var]["robin"]["x"].shape[0] > 0
                ):
                    x_bc = global_boundary_dict[var]["robin"]["x"]
                    u_bc = global_boundary_dict[var]["robin"]["u"]
                    normals = global_boundary_dict[var]["robin"]["normals"]
                    params = global_boundary_dict[var]["robin"]["params"]

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

        # Combine losses with appropriate weights
        total_loss = pde_loss + boundary_loss_weight * boundary_loss

        # Occasionally print loss components for debugging
        if torch.rand(1).item() < 0.01:  # 1% chance of printing
            print(f"\nLoss Components:")
            print(f"Boundary Loss: {boundary_loss.item():.8f}")
            print(f"Total Loss: {total_loss.item():.8f}")

        return total_loss
