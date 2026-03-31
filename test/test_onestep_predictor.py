import numpy as np

from src.problem_solvers.nonlinear_pde_solver.time_schemes.onestep_predictor import (
    OneStepPredictor,
)


class DummySchemeConfig:
    n_eqs = 1


class DummyFitter:
    def __init__(self):
        self._features = [[np.ones((2, 2))]]
        self._linear_operators = {0: {"L1": np.zeros((1, 2, 2)), "L2": np.zeros((1, 2, 2))}}
        self.dgN = 2
        self.data = {
            "x": np.array([[0.0], [0.5], [1.0]]),
            "x_segments_norm": [np.array([[0.0], [1.0]])],
        }
        self.last_features_arg = None

    def F_func(self, features, U_mid):
        self.last_features_arg = features
        return np.full((U_mid.shape[0], 1), 2.0)

    def N_func(self, features, U_mid):
        self.last_features_arg = features
        return [np.full(2, 0.5)]


def build_onestep_predictor():
    scheme = OneStepPredictor.__new__(OneStepPredictor)
    scheme.config = DummySchemeConfig()
    scheme.fitter = DummyFitter()
    scheme.L1_beta_star = [[np.array([2.0, 2.0])]]
    scheme.L1_beta_prev = [[np.array([0.0, 0.0])]]
    scheme.L2_beta_star = [[np.array([3.0, 3.0])]]
    scheme.L2_beta_prev = [[np.array([1.0, 1.0])]]
    return scheme


def test_onestep_predictor_corrector_step_uses_operator_averages():
    scheme = build_onestep_predictor()
    U_n_seg = [np.array([[10.0], [20.0]])]
    U_star_seg = [np.array([[12.0], [22.0]])]
    coeffs_star = np.array([[[1.0, 2.0]]])

    U_new_seg, coeffs_new = scheme._corrector_step(
        U_n_seg,
        U_star_seg,
        dt=0.1,
        coeffs_star=coeffs_star,
        step=2,
    )

    expected = np.array([[9.45], [19.45]])
    assert np.allclose(U_new_seg[0], expected)
    assert coeffs_new is coeffs_star


def test_onestep_predictor_estimate_stable_dt_respects_bounds():
    scheme = build_onestep_predictor()
    dt = scheme.estimate_stable_dt(np.array([[10.0], [5.0]]), safety_factor=0.8)
    assert 1e-6 <= dt <= 0.1


def test_onestep_predictor_build_stage_jacobian_rejects_unknown_operation():
    scheme = build_onestep_predictor()
    try:
        scheme.build_stage_jacobian(0, operation="unsupported", dt=0.1)
    except ValueError as exc:
        assert "Only onestep_predictor needs Jacobian" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported operation")


def test_onestep_predictor_uses_base_feature_matrix_for_nonlinear_operators():
    scheme = build_onestep_predictor()
    U_n_seg = [np.array([[1.0], [2.0]])]

    scheme.build_stage_jacobian(0, operation="onestep_predictor", dt=0.1, U_n_seg=U_n_seg)

    assert isinstance(scheme.fitter.last_features_arg, np.ndarray)
    assert scheme.fitter.last_features_arg.shape == (2, 2)
