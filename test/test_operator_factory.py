import numpy as np

from src.abstract_class.operator_factory import create_operator_factory


def test_linear_operator_factory_returns_expected_matrix():
    all_derivatives = {"U_X": [0, 1]}
    operator_terms = {
        "L1": [
            {
                "derivative_indices": [1],
                "symbolic_expr": "2 * U_X",
                "var_idx": 0,
            }
        ]
    }
    features = [np.zeros((2, 2)), np.array([[1.0, 2.0], [3.0, 4.0]])]

    factory = create_operator_factory(all_derivatives, optimized=False)
    ops = factory.create_all_operators(operator_terms)
    result = ops["L1_func"](features)

    assert result.shape == (1, 2, 2)
    assert np.allclose(result[0], np.array([[2.0, 4.0], [6.0, 8.0]]))


def test_linear_operator_factory_optimized_matches_nonoptimized():
    all_derivatives = {"U_X": [0, 1]}
    operator_terms = {
        "L1": [
            {
                "derivative_indices": [1],
                "symbolic_expr": "U_X",
                "var_idx": 0,
            }
        ]
    }
    features = [np.zeros((2, 2)), np.array([[1.0, 2.0], [3.0, 4.0]])]

    basic_factory = create_operator_factory(all_derivatives, optimized=False)
    optimized_factory = create_operator_factory(all_derivatives, optimized=True)

    basic_result = basic_factory.create_all_operators(operator_terms)["L1_func"](features)
    optimized_result = optimized_factory.create_all_operators(operator_terms)["L1_func"](features)

    assert np.allclose(basic_result, optimized_result)


def test_nonlinear_operator_factory_uses_u_for_zeroth_order_terms():
    all_derivatives = {"U": [0, 0]}
    operator_terms = {
        "N": [
            {
                "derivative_indices": [0],
                "symbolic_expr": "U + 1",
                "var_idx": 0,
            }
        ]
    }
    u = np.array([[2.0], [5.0]])

    factory = create_operator_factory(all_derivatives, optimized=False)
    ops = factory.create_all_operators(operator_terms)
    result = ops["N_func"](features=[np.zeros((2, 1))], u=u, coeffs=None, segment_idx=0)

    assert len(result) == 1
    assert np.allclose(result[0], np.array([3.0, 6.0]))


def test_f_operator_factory_formats_output_matrix():
    all_derivatives = {"U": [0, 0]}
    operator_terms = {
        "F": [
            {
                "derivative_indices": [0],
                "symbolic_expr": "U",
                "var_idx": 0,
            }
        ]
    }
    u = np.array([[2.0], [5.0]])

    factory = create_operator_factory(all_derivatives, optimized=False)
    ops = factory.create_all_operators(operator_terms)
    result = ops["F_func"](features=[np.zeros((2, 1))], u=u, coeffs=None, segment_idx=0)

    assert result.shape == (2, 1)
    assert np.allclose(result[:, 0], np.array([2.0, 5.0]))
