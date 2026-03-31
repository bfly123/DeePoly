"""Matrix/result shape helpers for operator outputs and linear algebra plumbing."""

import numpy as np


def validate_operator_output(
    result: np.ndarray,
    expected_shape: tuple,
    name: str = "operator_result",
) -> np.ndarray:
    """Validate an operator output and reshape only when size matches exactly."""
    if result.shape == expected_shape:
        return result
    if result.size == np.prod(expected_shape):
        return result.reshape(expected_shape)
    raise ValueError(f"{name} has shape {result.shape}, expected {expected_shape}")


def ensure_matrix_result(
    result: np.ndarray,
    n_points: int,
    n_eqs: int,
    name: str = "matrix_result",
) -> np.ndarray:
    """Ensure a matrix/vector result matches the expected point/equation layout."""
    expected_shape = (n_points, n_eqs)

    if result.ndim == 1:
        if result.size == n_points:
            if n_eqs == 1:
                return result.reshape(-1, 1)
            raise ValueError(f"{name} is 1D but {n_eqs} equations expected")
        if result.size == n_points * n_eqs:
            return result.reshape(expected_shape)
        raise ValueError(
            f"{name} size {result.size} doesn't match expected {n_points}x{n_eqs}"
        )

    if result.ndim == 2:
        if result.shape == expected_shape:
            return result
        raise ValueError(f"{name} has shape {result.shape}, expected {expected_shape}")

    raise ValueError(f"{name} must be 1D or 2D, got {result.ndim}D")
