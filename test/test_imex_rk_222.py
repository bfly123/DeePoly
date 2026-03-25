import numpy as np

from src.problem_solvers.nonlinear_pde_solver.time_schemes.imex_rk_222 import ImexRK222


class DummyConfig:
    n_eqs = 2


def test_imex_rk_222_normalizes_nonlinear_operator_list_output():
    scheme = ImexRK222(DummyConfig())

    nonlinear_terms = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
    ]

    matrix = scheme._nonlinear_terms_to_matrix(nonlinear_terms, n_points=3)

    assert matrix.shape == (3, 2)
    assert np.allclose(matrix[:, 0], [1.0, 2.0, 3.0])
    assert np.allclose(matrix[:, 1], [4.0, 5.0, 6.0])


def test_imex_rk_222_build_stage_rhs_accepts_normalized_nonlinear_matrix():
    scheme = ImexRK222(DummyConfig())

    rhs = scheme._build_stage_rhs(
        0,
        U_n_seg=np.array([[10.0, 20.0], [30.0, 40.0]]),
        N_n=np.array([[1.0, 2.0], [3.0, 4.0]]),
        stage=1,
        dt=0.1,
    )

    expected = np.array(
        [
            [10.0 - scheme.gamma * 0.1 * 1.0, 20.0 - scheme.gamma * 0.1 * 2.0],
            [30.0 - scheme.gamma * 0.1 * 3.0, 40.0 - scheme.gamma * 0.1 * 4.0],
        ]
    )

    assert np.allclose(rhs, expected)


def test_imex_rk_222_stage_two_aliases_are_resolved():
    scheme = ImexRK222(DummyConfig())

    u_n_seg, u_prev_seg = scheme._resolve_stage_inputs(
        0,
        {
            "U_n_seg": [np.array([[1.0, 2.0]])],
            "U_1_seg": [np.array([[3.0, 4.0]])],
        },
    )
    beta_prev = scheme._resolve_previous_coefficients(
        0,
        {
            "coeffs_1": np.array([[[5.0, 6.0], [7.0, 8.0]]]),
        },
    )

    assert np.allclose(u_n_seg, [[1.0, 2.0]])
    assert np.allclose(u_prev_seg, [[3.0, 4.0]])
    assert np.allclose(beta_prev, [[5.0, 6.0], [7.0, 8.0]])


def test_imex_rk_222_stage_one_operator_contribution_includes_nonlinear_term():
    scheme = ImexRK222(DummyConfig())

    class DummyFitter:
        ns = 1
        dgN = 2
        _features = {0: [np.array([[1.0], [1.0]])]}
        _linear_operators = {
            0: {
                "L1": [np.zeros((2, 2)), np.zeros((2, 2))],
                "L2": [np.zeros((2, 2)), np.zeros((2, 2))],
            }
        }

        @staticmethod
        def F_func(_features_list, U_seg):
            return np.zeros_like(U_seg)

        @staticmethod
        def N_func(_features_list, _U_seg):
            return [
                np.array([1.5, 2.5]),
                np.array([3.5, 4.5]),
            ]

        @staticmethod
        def has_operator(_name):
            return True

    scheme.fitter = DummyFitter()

    contribution = scheme._build_segment_operator_contribution(
        segment_idx=0,
        U_seg=np.zeros((2, 2)),
        coeffs_stage=np.zeros((1, 2, 2)),
        stage=1,
        U_n_seg=[np.zeros((2, 2))],
        U_1_seg=[],
    )

    assert np.allclose(contribution, [[1.5, 3.5], [2.5, 4.5]])
