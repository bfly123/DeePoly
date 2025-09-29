"""
Unified Boundary Condition Processing System
Based on time_pde_solver's boundary condition handling as the foundation
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseBoundaryConditionProcessor(ABC):
    """Base class for unified boundary condition processing"""

    def __init__(self, config):
        self.config = config
        self.boundary_loss_weight = 10.0

    def compute_boundary_loss(self, data_GPU: Dict, network) -> torch.Tensor:
        """Unified boundary condition loss computation - eliminate type branching

        Args:
            data_GPU: GPU data dictionary containing boundary conditions
            network: Neural network instance for predictions

        Returns:
            torch.Tensor: Total boundary loss
        """
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
                    total_boundary_loss += handler(var_bc_dict[bc_type], network)

        return total_boundary_loss

    def _has_valid_bc_data(self, bc_data: Dict) -> bool:
        """Unified boundary condition data validation - eliminate type-specific checks

        Args:
            bc_data: Boundary condition data dictionary

        Returns:
            bool: True if data is valid
        """
        if not bc_data:
            return False

        # Check for standard data structure
        if "x" in bc_data:
            return bc_data["x"].shape[0] > 0

        # Check for periodic-specific structure
        if "pairs" in bc_data:
            return bool(bc_data["pairs"])

        return False

    def _compute_dirichlet_loss(self, bc_data: Dict, network) -> torch.Tensor:
        """Compute Dirichlet boundary condition loss

        Args:
            bc_data: Dirichlet boundary condition data
            network: Neural network instance

        Returns:
            torch.Tensor: Dirichlet loss value
        """
        x_bc = bc_data["x"]
        u_bc = bc_data["values"]

        _, pred_bc = network(x_bc)
        return torch.mean((pred_bc - u_bc) ** 2)

    def _compute_neumann_loss(self, bc_data: Dict, network) -> torch.Tensor:
        """Compute Neumann boundary condition loss

        Args:
            bc_data: Neumann boundary condition data
            network: Neural network instance

        Returns:
            torch.Tensor: Neumann loss value
        """
        x_bc = bc_data["x"]
        u_bc = bc_data["values"]
        normals = bc_data["normals"]

        errors = []
        for i in range(x_bc.shape[0]):
            x_point = x_bc[i:i+1].clone().detach().requires_grad_(True)
            _, u_pred = network(x_point)

            grads = network.gradients(u_pred, x_point)[0]
            normal_derivative = torch.sum(grads * normals[i])

            errors.append((normal_derivative - u_bc[i]) ** 2)

        return torch.mean(torch.stack(errors)) if errors else torch.tensor(0.0, device=network.config.device)

    def _compute_robin_loss(self, bc_data: Dict, network) -> torch.Tensor:
        """Compute Robin boundary condition loss

        Args:
            bc_data: Robin boundary condition data
            network: Neural network instance

        Returns:
            torch.Tensor: Robin loss value
        """
        x_bc = bc_data["x"]
        u_bc = bc_data["values"]
        normals = bc_data["normals"]
        params = bc_data["params"]
        alpha, beta = params[0], params[1]

        errors = []
        for i in range(x_bc.shape[0]):
            x_point = x_bc[i:i+1].clone().detach().requires_grad_(True)
            _, u_pred = network(x_point)

            if abs(beta) > 1e-10:
                grads = network.gradients(u_pred, x_point)[0]
                normal_derivative = torch.sum(grads * normals[i])
                robin_value = alpha * u_pred + beta * normal_derivative
            else:
                robin_value = alpha * u_pred

            errors.append((robin_value - u_bc[i]) ** 2)

        return torch.mean(torch.stack(errors)) if errors else torch.tensor(0.0, device=network.config.device)

    def _compute_periodic_loss(self, periodic_data: Dict, network) -> torch.Tensor:
        """Unified periodic boundary condition loss computation

        Args:
            periodic_data: Periodic boundary condition data
            network: Neural network instance

        Returns:
            torch.Tensor: Periodic loss value
        """
        total_loss = 0.0

        for pair in periodic_data['pairs']:
            x_bc_1, x_bc_2 = pair['x_1'], pair['x_2']

            _, pred_bc_1 = network(x_bc_1)
            _, pred_bc_2 = network(x_bc_2)

            # Unified periodic boundary condition: U(x1) = U(x2)
            total_loss += torch.mean((pred_bc_1 - pred_bc_2) ** 2)

        return total_loss

    def prepare_boundary_gpu_data(self, data: Dict) -> Dict:
        """Prepare boundary condition data for GPU processing

        Args:
            data: Input data dictionary containing boundary conditions

        Returns:
            Dict: GPU-ready boundary condition data
        """
        # Transfer global_boundary_dict data to GPU - Pure abstract U processing
        if "global_boundary_dict" not in data:
            return {"global_boundary_dict": {}}

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
                                        pair_value, dtype=torch.float64,
                                        device=self.config.device, requires_grad=True
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
                                    value, dtype=torch.float64,
                                    device=self.config.device, requires_grad=True
                                )
                            else:
                                global_boundary_dict[var_idx][bc_type][key] = torch.tensor(
                                    value, dtype=torch.float64, device=self.config.device
                                )
                        else:
                            global_boundary_dict[var_idx][bc_type][key] = value

        return {"global_boundary_dict": global_boundary_dict}


class TimePDEBoundaryProcessor(BaseBoundaryConditionProcessor):
    """Boundary condition processor for time-dependent PDEs"""
    pass


class LinearPDEBoundaryProcessor(BaseBoundaryConditionProcessor):
    """Boundary condition processor for linear PDEs"""
    pass


class FuncFittingBoundaryProcessor(BaseBoundaryConditionProcessor):
    """Boundary condition processor for function fitting problems"""
    pass


def create_boundary_processor(problem_type: str, config) -> BaseBoundaryConditionProcessor:
    """Factory function to create appropriate boundary processor

    Args:
        problem_type: Type of problem ('time_pde', 'linear_pde', 'func_fitting')
        config: Configuration object

    Returns:
        BaseBoundaryConditionProcessor: Appropriate boundary processor instance
    """
    processors = {
        'time_pde': TimePDEBoundaryProcessor,
        'linear_pde': LinearPDEBoundaryProcessor,
        'func_fitting': FuncFittingBoundaryProcessor
    }

    if problem_type not in processors:
        raise ValueError(f"Unknown problem type: {problem_type}")

    return processors[problem_type](config)


# Mixin class for neural networks to use unified boundary processing
class BoundaryConditionMixin:
    """Mixin class to add unified boundary condition processing to neural networks"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize boundary processor based on problem type
        problem_type = getattr(self.config, 'problem_type', 'linear_pde')
        self.boundary_processor = create_boundary_processor(problem_type, self.config)

    def _compute_boundary_loss(self, data_GPU: Dict) -> torch.Tensor:
        """Unified boundary loss computation using the boundary processor

        Args:
            data_GPU: GPU data dictionary

        Returns:
            torch.Tensor: Boundary loss value
        """
        return self.boundary_processor.compute_boundary_loss(data_GPU, self)

    def prepare_boundary_gpu_data(self, data: Dict) -> Dict:
        """Prepare boundary GPU data using the boundary processor

        Args:
            data: Input data dictionary

        Returns:
            Dict: GPU-ready boundary data
        """
        return self.boundary_processor.prepare_boundary_gpu_data(data)