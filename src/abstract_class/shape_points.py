"""Point/equation shape helpers used across fitters, operators, and data pipelines."""

from typing import Optional

import numpy as np


def ensure_points_eqs(u: np.ndarray, name: str = "u") -> np.ndarray:
    """Ensure an array uses the canonical ``(n_points, n_eqs)`` layout."""
    if u.ndim == 1:
        return u.reshape(-1, 1)
    if u.ndim == 2:
        return u
    raise ValueError(f"{name} must be 1D or 2D, got {u.ndim}D with shape {u.shape}")


def safe_eq_col(u: np.ndarray, eq_idx: int, name: str = "u") -> np.ndarray:
    """Extract one equation column while preserving 2D shape."""
    u = ensure_points_eqs(u, name)
    if eq_idx >= u.shape[1]:
        raise ValueError(
            f"Equation index {eq_idx} out of bounds for {name} "
            f"with {u.shape[1]} equations"
        )
    return u[:, [eq_idx]]


def safe_all_eqs(u: np.ndarray, name: str = "u") -> np.ndarray:
    """Return all equations in standardized 2D form."""
    return ensure_points_eqs(u, name)


def assert_points_eqs(u: np.ndarray, name: str = "u") -> None:
    """Assert an array is already shaped as ``(n_points, n_eqs)``."""
    if u.ndim != 2:
        raise AssertionError(
            f"{name} must be 2D with shape (n_points, n_eqs), got shape {u.shape}"
        )


def standardize_solution_shape(
    u: np.ndarray,
    n_eqs: Optional[int] = None,
    name: str = "u",
) -> np.ndarray:
    """Standardize a solution array to ``(n_points, n_eqs)`` format."""
    u_std = ensure_points_eqs(u, name)
    if n_eqs is None or u_std.shape[1] == n_eqs:
        return u_std
    if u_std.shape[1] == 1 and n_eqs > 1:
        return np.tile(u_std, (1, n_eqs))
    raise ValueError(f"{name} has {u_std.shape[1]} equations, expected {n_eqs}")


def safe_segment_slice(
    u_global: np.ndarray,
    start_idx: int,
    end_idx: int,
    name: str = "u_global",
) -> np.ndarray:
    """Slice a global solution while preserving canonical 2D shape."""
    u_std = ensure_points_eqs(u_global, name)
    return u_std[start_idx:end_idx, :].copy()


def concat_segments(u_segments: list, name: str = "u_segments") -> np.ndarray:
    """Concatenate segment-local solutions into one global solution array."""
    if not u_segments:
        raise ValueError(f"{name} cannot be empty")

    standardized = [ensure_points_eqs(seg, f"{name}[{i}]") for i, seg in enumerate(u_segments)]
    n_eqs = standardized[0].shape[1]
    for index, seg in enumerate(standardized[1:], 1):
        if seg.shape[1] != n_eqs:
            raise ValueError(
                f"{name}[{index}] has {seg.shape[1]} equations, expected {n_eqs}"
            )
    return np.vstack(standardized)
