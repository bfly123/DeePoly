"""Coefficient-shape helpers for DeePoly linear systems and basis assembly."""

import numpy as np


def broadcast_coeffs(
    coeffs: np.ndarray,
    target_shape: tuple,
    name: str = "coeffs",
) -> np.ndarray:
    """Normalize coefficients to a target segment/equation/basis rank."""
    if coeffs.ndim == 1:
        if len(target_shape) == 1:
            return coeffs
        if len(target_shape) == 2:
            return coeffs.reshape(1, -1)
        if len(target_shape) == 3:
            return coeffs.reshape(1, 1, -1)
        raise ValueError(f"Cannot broadcast 1D {name} to shape {target_shape}")

    if coeffs.ndim == 2:
        if len(target_shape) == 2:
            return coeffs
        if len(target_shape) == 3:
            return coeffs.reshape(1, coeffs.shape[0], coeffs.shape[1])
        raise ValueError(f"Cannot broadcast 2D {name} to shape {target_shape}")

    if coeffs.ndim == 3:
        return coeffs

    raise ValueError(
        f"{name} must be 1D, 2D, or 3D, got {coeffs.ndim}D with shape {coeffs.shape}"
    )
