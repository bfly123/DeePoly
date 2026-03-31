"""
Unified Boundary Condition Processing System
Based on time_pde_solver's boundary condition handling as the foundation
"""

from typing import Dict
from abc import ABC, abstractmethod

import torch

from src.abstract_class.boundary_condition_gpu_mixin import BoundaryConditionGpuMixin
from src.abstract_class.boundary_condition_loss_mixin import BoundaryConditionLossMixin


class BaseBoundaryConditionProcessor(
    BoundaryConditionLossMixin,
    BoundaryConditionGpuMixin,
    ABC,
):
    """Base class for unified boundary condition processing"""

    def __init__(self, config):
        self.config = config
        self.boundary_loss_weight = 10.0


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
        'func_fitting': FuncFittingBoundaryProcessor,
        'general_pde': TimePDEBoundaryProcessor,  # Reuse time_pde processor for general_pde
        'fixiteration_pde': TimePDEBoundaryProcessor  # Reuse time_pde processor for fixiteration_pde
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
