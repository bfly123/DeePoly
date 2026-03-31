import numpy as np
import pytest

from src.abstract_class.shape import (
    assert_points_eqs,
    broadcast_coeffs,
    concat_segments,
    ensure_matrix_result,
    ensure_points_eqs,
    get_equation_column,
    safe_segment_slice,
    standardize_solution_shape,
    validate_operator_output,
)


def test_ensure_points_eqs_and_aliases_keep_2d_layout():
    values = np.array([1.0, 2.0, 3.0])

    standardized = ensure_points_eqs(values, "u")
    column = get_equation_column(np.column_stack([values, values + 1.0]), 1, "u")

    assert standardized.shape == (3, 1)
    assert np.allclose(standardized[:, 0], values)
    assert column.shape == (3, 1)
    assert np.allclose(column[:, 0], values + 1.0)


def test_standardize_solution_shape_broadcasts_single_equation_when_requested():
    values = np.array([1.0, 2.0, 3.0])

    standardized = standardize_solution_shape(values, n_eqs=2, name="u")

    assert standardized.shape == (3, 2)
    assert np.allclose(standardized[:, 0], values)
    assert np.allclose(standardized[:, 1], values)


def test_concat_segments_and_safe_segment_slice_preserve_equation_count():
    segments = [
        np.array([1.0, 2.0]),
        np.array([[3.0], [4.0]]),
    ]

    combined = concat_segments(segments, "u_segments")
    sliced = safe_segment_slice(combined, 1, 3, "u_global")

    assert combined.shape == (4, 1)
    assert np.allclose(combined[:, 0], [1.0, 2.0, 3.0, 4.0])
    assert sliced.shape == (2, 1)
    assert np.allclose(sliced[:, 0], [2.0, 3.0])


def test_broadcast_coeffs_handles_vector_and_matrix_targets():
    coeffs_1d = np.array([1.0, 2.0, 3.0])
    coeffs_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert broadcast_coeffs(coeffs_1d, (3,)).shape == (3,)
    assert broadcast_coeffs(coeffs_1d, (2, 3)).shape == (1, 3)
    assert broadcast_coeffs(coeffs_1d, (4, 2, 3)).shape == (1, 1, 3)
    assert broadcast_coeffs(coeffs_2d, (2, 2)).shape == (2, 2)
    assert broadcast_coeffs(coeffs_2d, (5, 2, 2)).shape == (1, 2, 2)


def test_validate_operator_output_and_ensure_matrix_result_reshape_when_legal():
    flat = np.arange(6.0)

    validated = validate_operator_output(flat, (2, 3), "result")
    matrix = ensure_matrix_result(flat, 2, 3, "matrix")

    assert validated.shape == (2, 3)
    assert matrix.shape == (2, 3)
    assert np.allclose(validated, matrix)


def test_shape_helpers_raise_clear_errors_for_invalid_shapes():
    with pytest.raises(AssertionError):
        assert_points_eqs(np.array([1.0, 2.0]), "u")

    with pytest.raises(ValueError):
        ensure_points_eqs(np.zeros((2, 2, 2)), "u")

    with pytest.raises(ValueError):
        validate_operator_output(np.arange(5.0), (2, 3), "result")

    with pytest.raises(ValueError):
        ensure_matrix_result(np.arange(4.0), 2, 3, "matrix")
