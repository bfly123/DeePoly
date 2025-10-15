"""
Shape utility functions for maintaining consistent (n_points, n_eqs) dimensions
throughout the DeePoly framework, eliminating single vs multi-equation branching logic.
"""
import numpy as np
from typing import Union, Optional


def ensure_points_eqs(u: np.ndarray, name: str = "u") -> np.ndarray:
    """
    Ensure array has shape (n_points, n_eqs).

    Args:
        u: Input array to standardize
        name: Variable name for error messages

    Returns:
        Array with shape (n_points, n_eqs)

    Raises:
        ValueError: If input is not 1D or 2D
    """
    if u.ndim == 1:
        return u.reshape(-1, 1)
    elif u.ndim == 2:
        return u
    else:
        raise ValueError(f"{name} must be 1D or 2D, got {u.ndim}D with shape {u.shape}")


def safe_eq_col(u: np.ndarray, eq_idx: int, name: str = "u") -> np.ndarray:
    """
    Extract equation column preserving 2D shape.

    Args:
        u: Input array with shape (n_points, n_eqs)
        eq_idx: Equation index to extract
        name: Variable name for error messages

    Returns:
        Array with shape (n_points, 1)
    """
    u = ensure_points_eqs(u, name)
    if eq_idx >= u.shape[1]:
        raise ValueError(f"Equation index {eq_idx} out of bounds for {name} with {u.shape[1]} equations")
    return u[:, [eq_idx]]  # Returns (n_points, 1)


def safe_all_eqs(u: np.ndarray, name: str = "u") -> np.ndarray:
    """
    Get all equations in standardized 2D format.

    Args:
        u: Input array
        name: Variable name for error messages

    Returns:
        Array with shape (n_points, n_eqs)
    """
    return ensure_points_eqs(u, name)


def broadcast_coeffs(coeffs: np.ndarray, target_shape: tuple, name: str = "coeffs") -> np.ndarray:
    """
    Normalize coefficients to target shape with consistent indexing.

    Args:
        coeffs: Input coefficient array
        target_shape: Target shape tuple (n_segments, n_eqs, n_basis) or subset
        name: Variable name for error messages

    Returns:
        Coefficients array with standardized shape
    """
    if coeffs.ndim == 1:
        # Single coefficient vector - broadcast to required dimensions
        if len(target_shape) == 1:
            return coeffs
        elif len(target_shape) == 2:
            return coeffs.reshape(1, -1)
        elif len(target_shape) == 3:
            return coeffs.reshape(1, 1, -1)
        else:
            raise ValueError(f"Cannot broadcast 1D {name} to shape {target_shape}")

    elif coeffs.ndim == 2:
        if len(target_shape) == 2:
            return coeffs
        elif len(target_shape) == 3:
            return coeffs.reshape(1, coeffs.shape[0], coeffs.shape[1])
        else:
            raise ValueError(f"Cannot broadcast 2D {name} to shape {target_shape}")

    elif coeffs.ndim == 3:
        return coeffs

    else:
        raise ValueError(f"{name} must be 1D, 2D, or 3D, got {coeffs.ndim}D with shape {coeffs.shape}")


def assert_points_eqs(u: np.ndarray, name: str = "u") -> None:
    """
    Assert array has 2D shape (n_points, n_eqs).

    Args:
        u: Array to check
        name: Variable name for error messages

    Raises:
        AssertionError: If array is not 2D
    """
    if u.ndim != 2:
        raise AssertionError(f"{name} must be 2D with shape (n_points, n_eqs), got shape {u.shape}")


def standardize_solution_shape(u: np.ndarray, n_eqs: Optional[int] = None, name: str = "u") -> np.ndarray:
    """
    Standardize solution array to (n_points, n_eqs) format.

    Args:
        u: Input solution array
        n_eqs: Expected number of equations (optional)
        name: Variable name for error messages

    Returns:
        Standardized solution array with shape (n_points, n_eqs)
    """
    u_std = ensure_points_eqs(u, name)

    if n_eqs is not None and u_std.shape[1] != n_eqs:
        if u_std.shape[1] == 1 and n_eqs > 1:
            # Broadcast single equation to multi-equation format
            u_std = np.tile(u_std, (1, n_eqs))
        else:
            raise ValueError(f"{name} has {u_std.shape[1]} equations, expected {n_eqs}")

    return u_std


def safe_segment_slice(u_global: np.ndarray, start_idx: int, end_idx: int, name: str = "u_global") -> np.ndarray:
    """
    Extract segment slice maintaining 2D shape.

    Args:
        u_global: Global solution array
        start_idx: Start index for segment
        end_idx: End index for segment
        name: Variable name for error messages

    Returns:
        Segment array with shape (n_points_seg, n_eqs)
    """
    u_std = ensure_points_eqs(u_global, name)
    return u_std[start_idx:end_idx, :].copy()


def concat_segments(u_segments: list, name: str = "u_segments") -> np.ndarray:
    """
    Concatenate segment solutions maintaining 2D shape.

    Args:
        u_segments: List of segment solution arrays
        name: Variable name for error messages

    Returns:
        Global solution array with shape (n_points_total, n_eqs)
    """
    if not u_segments:
        raise ValueError(f"{name} cannot be empty")

    # Standardize all segments
    u_std_segments = [ensure_points_eqs(seg, f"{name}[{i}]") for i, seg in enumerate(u_segments)]

    # Check consistent equation count
    n_eqs = u_std_segments[0].shape[1]
    for i, seg in enumerate(u_std_segments[1:], 1):
        if seg.shape[1] != n_eqs:
            raise ValueError(f"{name}[{i}] has {seg.shape[1]} equations, expected {n_eqs}")

    return np.vstack(u_std_segments)


def validate_operator_output(result: np.ndarray, expected_shape: tuple, name: str = "operator_result") -> np.ndarray:
    """
    Validate and standardize operator output shape.

    Args:
        result: Operator result array
        expected_shape: Expected shape (n_points, n_eqs) or similar
        name: Variable name for error messages

    Returns:
        Validated result array
    """
    if result.shape != expected_shape:
        # Try to reshape if possible
        if result.size == np.prod(expected_shape):
            result = result.reshape(expected_shape)
        else:
            raise ValueError(f"{name} has shape {result.shape}, expected {expected_shape}")

    return result


def ensure_matrix_result(result: np.ndarray, n_points: int, n_eqs: int, name: str = "matrix_result") -> np.ndarray:
    """
    Ensure matrix operation result has correct 2D shape.

    Args:
        result: Matrix operation result
        n_points: Expected number of points
        n_eqs: Expected number of equations
        name: Variable name for error messages

    Returns:
        Result array with shape (n_points, n_eqs)
    """
    expected_shape = (n_points, n_eqs)

    if result.ndim == 1:
        if result.size == n_points:
            # Single equation result
            if n_eqs == 1:
                return result.reshape(-1, 1)
            else:
                raise ValueError(f"{name} is 1D but {n_eqs} equations expected")
        elif result.size == n_points * n_eqs:
            return result.reshape(expected_shape)
        else:
            raise ValueError(f"{name} size {result.size} doesn't match expected {n_points}×{n_eqs}")

    elif result.ndim == 2:
        if result.shape == expected_shape:
            return result
        else:
            raise ValueError(f"{name} has shape {result.shape}, expected {expected_shape}")

    else:
        raise ValueError(f"{name} must be 1D or 2D, got {result.ndim}D")


# Backward compatibility aliases
standardize_array = ensure_points_eqs
get_equation_column = safe_eq_col