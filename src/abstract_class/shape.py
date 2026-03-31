"""Compatibility facade for shape helpers grouped by responsibility."""

from .shape_coeffs import broadcast_coeffs
from .shape_matrix import ensure_matrix_result, validate_operator_output
from .shape_points import (
    assert_points_eqs,
    concat_segments,
    ensure_points_eqs,
    safe_all_eqs,
    safe_eq_col,
    safe_segment_slice,
    standardize_solution_shape,
)

# Backward compatibility aliases
standardize_array = ensure_points_eqs
get_equation_column = safe_eq_col

__all__ = [
    "assert_points_eqs",
    "broadcast_coeffs",
    "concat_segments",
    "ensure_matrix_result",
    "ensure_points_eqs",
    "get_equation_column",
    "safe_all_eqs",
    "safe_eq_col",
    "safe_segment_slice",
    "standardize_array",
    "standardize_solution_shape",
    "validate_operator_output",
]
