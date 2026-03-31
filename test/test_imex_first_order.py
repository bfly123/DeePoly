import numpy as np

from src.problem_solvers.nonlinear_pde_solver.time_schemes.imex_1st import ImexFirstOrder


class DummyConfig:
    n_eqs = 1


class DummyFitter:
    def __init__(self):
        self.data = {"x_segments_norm": [np.array([[0.0], [1.0]])]}
        self.dgN = 2
        self.ns = 1
        self._precompiled = True
        self._linear_operators = [
            {
                "L1": [np.array([[2.0, 0.0], [0.0, 2.0]])],
                "L2": [np.array([[1.0, 0.0], [0.0, 1.0]])],
            }
        ]
        self._features = [[np.array([[1.0, 0.0], [0.0, 1.0]])]]
        self._current_model = object()

    def fit(self, **kwargs):
        assert kwargs["operation"] == "imex_stage"
        return np.array([[1.0], [2.0]])

    def construct(self, data, model, coeffs_stage):
        del data, model
        return coeffs_stage, [coeffs_stage]

    def segments_to_global(self, segments):
        return segments[0]

    def F_func(self, features, U_n_seg):
        del features, U_n_seg
        return np.array([[3.0], [4.0]])

    def N_func(self, features, U_n_seg):
        del features, U_n_seg
        return [np.array([0.5, 1.0])]

    def has_operator(self, name):
        return name in {"L1", "L2", "N", "F"}


def build_scheme():
    scheme = ImexFirstOrder(DummyConfig())
    scheme.fitter = DummyFitter()
    return scheme


def test_imex_first_order_time_step_and_butcher_tableau():
    scheme = build_scheme()
    U_new, U_seg_new, coeffs_new = scheme.time_step(
        U_n=np.array([[0.0], [0.0]]),
        U_seg=[np.array([[1.0], [2.0]])],
        dt=0.1,
        current_time=0.2,
        step=3,
    )
    assert np.allclose(U_new, [[1.0], [2.0]])
    assert np.allclose(U_seg_new[0], [[1.0], [2.0]])
    assert np.allclose(coeffs_new, [[1.0], [2.0]])
    assert scheme.get_butcher_tableau()["gamma"] == 1.0


def test_imex_first_order_build_stage_jacobian_and_rhs():
    scheme = build_scheme()
    L_final, b_vector = scheme.build_stage_jacobian(
        0,
        dt=0.1,
        U_n_seg=[np.array([[2.0], [3.0]])],
    )
    expected_j = np.array([[1.5, 0.0], [0.0, 1.6]])
    assert np.allclose(L_final, expected_j)
    assert np.allclose(b_vector, [1.95, 2.9])


def test_imex_first_order_validate_and_estimate_dt():
    scheme = build_scheme()
    validation = scheme.validate_operators()
    assert validation["imex_ready"] is True
    dt = scheme.estimate_stable_dt(np.array([[1.0], [2.0]]), safety_factor=0.8)
    assert dt > 0
    info = scheme.get_scheme_info()
    assert info["order"] == 1
