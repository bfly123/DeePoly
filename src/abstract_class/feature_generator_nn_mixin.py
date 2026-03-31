from typing import TYPE_CHECKING, Any, List, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _import_torch():
    import torch
    from torch import nn

    return torch, nn


class FeatureGeneratorNNMixin:
    @staticmethod
    def spotter_features(
        x: np.ndarray,
        x_min: Union[float, np.ndarray],
        x_max: Union[float, np.ndarray],
        model,
        derivative: List[int] = None,
        device: Any = "cuda",
    ) -> np.ndarray:
        """Calculate neural network features and their derivatives."""
        derivative = FeatureGeneratorNNMixin._normalize_derivative_input(x.shape[1], derivative)
        if FeatureGeneratorNNMixin._is_feature_provider(model):
            if np.all(derivative == 0):
                return model.hidden_features(x, x_min, x_max)
            return model.hidden_feature_derivatives(x, x_min, x_max, derivative)
        x_tensor = FeatureGeneratorNNMixin._build_input_tensor(x, x_min, x_max, device)
        h, _ = model(x_tensor)
        if np.all(derivative == 0):
            return FeatureGeneratorNNMixin._tensor_to_numpy(h)
        return FeatureGeneratorNNMixin._compute_feature_derivatives(h, x_tensor, derivative)

    @staticmethod
    def _is_feature_provider(model_or_provider) -> bool:
        return all(
            hasattr(model_or_provider, attr_name)
            for attr_name in ("hidden_features", "hidden_feature_derivatives")
        )

    @staticmethod
    def _normalize_derivative_input(n_dims: int, derivative) -> np.ndarray:
        if derivative is None:
            derivative = [0] * n_dims
        if isinstance(derivative, int):
            derivative = [derivative] * n_dims
        derivative = np.asarray(derivative, dtype=int)
        if derivative.size != n_dims:
            derivative = np.resize(derivative, n_dims)
        if np.any(derivative < 0):
            raise ValueError("derivative orders cannot be negative")
        return derivative

    @staticmethod
    def _build_input_tensor(
        x: np.ndarray,
        x_min: Union[float, np.ndarray],
        x_max: Union[float, np.ndarray],
        device: Any,
    ):
        torch, _ = _import_torch()
        x = np.asarray(x, dtype=np.float64)
        x = x * (x_max - x_min) + x_min
        return torch.tensor(x, requires_grad=True, dtype=torch.float64).to(device)

    @staticmethod
    def _compute_feature_derivatives(
        h,
        x_tensor,
        derivative: np.ndarray,
    ) -> np.ndarray:
        jacobian = np.zeros((h.shape[0], h.shape[1]), dtype=np.float64)
        for feature_idx in range(h.shape[1]):
            current_grad = h[:, feature_idx]
            for dim, order in enumerate(derivative):
                current_grad = FeatureGeneratorNNMixin._apply_dimension_derivative(
                    current_grad,
                    x_tensor,
                    dim,
                    order,
                )
            jacobian[:, feature_idx] = FeatureGeneratorNNMixin._tensor_to_numpy(current_grad).squeeze()
        return jacobian

    @staticmethod
    def _apply_dimension_derivative(
        current_grad,
        x_tensor,
        dim: int,
        order: int,
    ):
        torch, _ = _import_torch()
        for derivative_idx in range(order):
            grad_tuple = torch.autograd.grad(
                outputs=current_grad,
                inputs=x_tensor,
                grad_outputs=torch.ones_like(current_grad),
                create_graph=True,
                retain_graph=True,
            )
            if grad_tuple is None or grad_tuple[0] is None:
                raise ValueError(
                    f"cannot compute {dim} derivative (current order {derivative_idx + 1}/{order})."
                )
            current_grad = grad_tuple[0][:, dim : dim + 1]
        return current_grad

    @staticmethod
    def init_coefficients(model, device: Any = "cuda"):
        """Extract the weight matrix from the last linear layer."""
        _, nn = _import_torch()
        model = model.to(device)
        last_layer = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_layer = module
        if last_layer is None:
            raise ValueError("Model中未FindLinear层")
        return FeatureGeneratorNNMixin._tensor_to_numpy(last_layer.weight)

    @staticmethod
    def _tensor_to_numpy(tensor) -> np.ndarray:
        tensor = tensor.detach().cpu()
        try:
            return tensor.numpy()
        except RuntimeError:
            return np.asarray(tensor.tolist(), dtype=np.float64)
