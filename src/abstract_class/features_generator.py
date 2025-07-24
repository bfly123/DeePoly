import numpy as np
import torch
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass
from torch import nn
from src.algebraic_solver.linear_solver import LinearSolver
import itertools
import math


class FeatureGenerator:
    """Feature generator class for polynomial and neural network features"""

    def __init__(self, device: torch.device):
        self.device = device

    @staticmethod
    def sniper_features(
        x: np.ndarray, degree: List[int], derivative: List[int] = None
    ) -> np.ndarray:
        """Generate polynomial features for arbitrary dimensions

        Args:
            x: Input data with shape (n_samples, n_dims)
            degree: List of polynomial degrees for each dimension
            derivative: List of derivative orders for each dimension, defaults to all zeros

        Returns:
            Feature matrix
        """
        n_dims = x.shape[1]

        # Handle input parameters
        if isinstance(degree, int):
            degree = [degree] * n_dims
        if derivative is None:
            derivative = [0] * n_dims
        if isinstance(derivative, int):
            derivative = [derivative] * n_dims

        # Validate inputs
        if len(degree) != n_dims or len(derivative) != n_dims:
            raise ValueError(f"degree and derivative must be lists of length {n_dims}")
        if any(d < 0 for d in derivative) or any(d < 0 for d in degree):
            raise ValueError("degrees and derivative orders cannot be negative")
        if any(derivative[i] > degree[i] for i in range(n_dims)):
            raise ValueError("derivative order cannot exceed polynomial degree")

        x = np.asarray(x, dtype=np.float64)

        # More efficient computation when no derivatives are needed
        if np.all(np.array(derivative) == 0):
            powers = [range(d + 1) for d in degree]
            terms = []
            for exponents in itertools.product(*powers):
                term = np.ones((x.shape[0], 1))
                for dim, power in enumerate(exponents):
                    if power > 0:  # Skip power of 0 for efficiency
                        term *= x[:, dim : dim + 1] ** power
                terms.append(term)
            return np.hstack(terms)

        # Handle derivatives
        powers = [range(d + 1) for d in degree]
        terms = []

        for exponents in itertools.product(*powers):
            if all(exp >= der for exp, der in zip(exponents, derivative)):
                # Calculate coefficient
                coef = 1.0
                for dim, (exp, der) in enumerate(zip(exponents, derivative)):
                    if der > 0:
                        coef *= math.factorial(exp) / math.factorial(exp - der)

                # Calculate term
                term = np.ones((x.shape[0], 1)) * coef
                for dim, (exp, der) in enumerate(zip(exponents, derivative)):
                    if exp - der > 0:
                        term *= x[:, dim : dim + 1] ** (exp - der)
            else:
                term = np.zeros((x.shape[0], 1))

            terms.append(term)

        return np.hstack(terms)

    @staticmethod
    def spotter_features(
        x: np.ndarray,
        x_min: Union[float, np.ndarray],
        x_max: Union[float, np.ndarray],
        model: torch.nn.Module,
        derivative: List[int] = None,
        device: torch.device = "cuda",
    ) -> np.ndarray:
        """Calculate neural network features and their derivatives

        Args:
            x: Input data with shape (n_samples, n_dims)
            x_min: Minimum values of input data
            x_max: Maximum values of input data
            model: Neural network model
            derivative: List of derivative orders for each dimension, defaults to all zeros
            device: Computation device

        Returns:
            Feature matrix or its derivatives
        """
        n_dims = x.shape[1]

        # Handle derivative parameters
        if derivative is None:
            derivative = [0] * n_dims
        if isinstance(derivative, int):
            derivative = [derivative] * n_dims

        # Validate inputs
        if len(derivative) != n_dims:
            raise ValueError(f"derivative must be a list of length {n_dims}")
        if any(d < 0 for d in derivative):
            raise ValueError("derivative orders cannot be negative")

        # Normalize input data
        x = x * (x_max - x_min) + x_min

        # Convert to tensor
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float64).to(device)

        # Get model output
        h, _ = model(x_tensor)

        # Return features directly if no derivatives are needed
        if all(d == 0 for d in derivative):
            return h.detach().cpu().numpy()

        # Calculate derivatives
        jacobian = np.zeros((h.shape[0], h.shape[1]), dtype=np.float64)
        for i in range(h.shape[1]):
            # Start with the i-th component of the NN output
            grad = h[:, i]

            # Apply derivatives iteratively based on the 'derivative' list
            # This assumes order matters, e.g., derivative=[1, 2] means d/dx (d^2/dy^2 f)
            current_grad = grad
            for dim, order in enumerate(derivative):
                if order > 0:
                    for _ in range(order):
                        grad_tuple = torch.autograd.grad(
                            outputs=current_grad,
                            inputs=x_tensor,
                            grad_outputs=torch.ones_like(current_grad),
                            create_graph=True,  # Essential for higher orders
                            retain_graph=True,  # Keep graph for next derivative calculation
                        )
                        if grad_tuple is None or grad_tuple[0] is None:
                            raise ValueError(
                                f"cannot compute {dim} derivative (current order {_ + 1}/{order})."
                            )

                        # Update current_grad to be the derivative w.r.t the current dimension
                        current_grad = grad_tuple[0][:, dim : dim + 1]

            # After applying all derivatives, store the result
            jacobian[:, i] = current_grad.detach().cpu().numpy().squeeze()

        return jacobian

    @staticmethod
    def init_coefficients(model: torch.nn.Module, device: torch.device = "cuda"):
        """
        Args:
            model: 神经网络模型
            device: 计算设备

        Returns:
            线性参数矩阵，维度为[ne, nn]，其中:
            - ne: 输出维度（方程数量）
            - nn: 导数第一个隐层维度
        """
        model = model.to(device)

        # 找到最后一个线性层
        last_layer = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_layer = module

        if last_layer is None:
            raise ValueError("模型中未找到线性层")

        # 提取权重矩阵 - PyTorch Linear层的权重形状为[out_features, in_features]
        # 这正好符合我们需要的[ne, nn]形状
        weight_matrix = last_layer.weight.detach().cpu().numpy()

        return weight_matrix
