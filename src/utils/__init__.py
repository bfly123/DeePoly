"""
Utility functions for the DeePoly framework.
"""

from .shape import (
    ensure_points_eqs,
    safe_eq_col,
    safe_all_eqs,
    broadcast_coeffs,
    assert_points_eqs,
    standardize_solution_shape,
    safe_segment_slice,
    concat_segments,
    validate_operator_output,
    ensure_matrix_result,
    # Backward compatibility
    standardize_array,
    get_equation_column
)

__all__ = [
    "ensure_points_eqs",
    "safe_eq_col",
    "safe_all_eqs",
    "broadcast_coeffs",
    "assert_points_eqs",
    "standardize_solution_shape",
    "safe_segment_slice",
    "concat_segments",
    "validate_operator_output",
    "ensure_matrix_result",
    "standardize_array",
    "get_equation_column"
]