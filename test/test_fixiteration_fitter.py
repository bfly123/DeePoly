import numpy as np

from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.core.fitter import (
    FixIterationPDEFitter,
)


class _FixConfig:
    n_dim = 1
    n_segments = [1]
    n_eqs = 1
    linear_device = "cpu"
    method = "poly"
    poly_degree = [1]
    DNN_degree = 0
    problem_type = "fixiteration_pde"
    const_list = []
    operator_terms = {}
    all_derivatives = {}
    max_iterations = 12
    residual_tol = 1e-6
    print_interval = 3


class _FixHarness(FixIterationPDEFitter):
    def __init__(self):
        self.config = _FixConfig()
        self.dgN = 2
        self.data = {"x_segments_norm": [np.zeros((3, 1))]}
        self._linear_operators = {0: {"L1": [np.ones((3, 2))], "L2": [np.full((3, 2), 2.0)]}}
        self._features = {0: [np.ones((3, 2))]}

    def global_to_segments(self, U_current):
        return [U_current]

    def F_func(self, features, u_seg):
        del features, u_seg
        return np.full((3, 1), 0.5)

    def N_func(self, features, u_seg):
        del features, u_seg
        return [np.array([1.0, 2.0, 3.0])]

    def _get_source_term(self, segment_idx):
        del segment_idx
        return np.array([[4.0], [5.0], [6.0]])


def test_fixiteration_fitter_initializes_runtime_defaults():
    fitter = FixIterationPDEFitter(config=_FixConfig(), data={})

    assert fitter.max_iterations == 12
    assert fitter.convergence_tol == 1e-6
    assert fitter.print_interval == 3
    assert fitter.residual_history == []


def test_fixiteration_segment_jacobian_builds_linearized_system():
    harness = _FixHarness()

    jacobian, rhs = harness._build_segment_jacobian(
        0,
        U_current=np.zeros((3, 1)),
    )

    expected_block = np.ones((3, 2)) + np.diag(np.full(3, 0.5)) @ np.full((3, 2), 2.0)
    assert jacobian.shape == (3, 2)
    assert np.allclose(jacobian, expected_block)
    assert np.allclose(rhs, np.array([3.0, 3.0, 3.0]))


def test_fixiteration_segment_jacobian_requires_current_solution():
    harness = _FixHarness()

    try:
        harness._build_segment_jacobian(0)
    except ValueError as exc:
        assert "U_current" in str(exc)
    else:
        raise AssertionError("Expected ValueError when U_current is missing")
