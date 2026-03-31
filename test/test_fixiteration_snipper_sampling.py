import numpy as np

from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver import (
    fixiteration_solver_factory_mixin as factory_module,
)
from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.fixiteration_solver_factory_mixin import (
    FixIterationSolverFactoryMixin,
)
from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.fixiteration_solver_output_mixin import (
    FixIterationSolverOutputMixin,
)
from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.fixiteration_solver_runtime_mixin import (
    FixIterationSolverRuntimeMixin,
)


class _RuntimeHarness(FixIterationSolverRuntimeMixin):
    def __init__(self):
        self.data_train = {"x": np.array([[1.0], [2.0]])}
        self.data_snipper = {"x": np.array([[3.0], [4.0], [5.0]])}
        self.captured_points = None

    def _get_iteration_dataset(self):
        return self.data_snipper

    def _predict_field_on_points(self, points):
        self.captured_points = np.asarray(points)
        return self.captured_points.copy()


class _VisualizerStub:
    def save_convergence_history(self, conv_info, results_dir):
        del conv_info, results_dir

    def plot_final_solution(self, U_test, data_test, save_path):
        del U_test, data_test, save_path

    def plot_residual_history(self, residual_history, steps, save_path):
        del residual_history, steps, save_path


class _OutputHarness(FixIterationSolverOutputMixin):
    def __init__(self):
        self.config = type("Config", (), {"results_dir": "/tmp"})()
        self.data_test = {"x": np.array([[9.0]])}
        self.data_snipper = {"x": np.array([[6.0], [7.0]])}
        self.visualizer = _VisualizerStub()
        self.interpolation_source = None
        self.saved = None

    def _get_iteration_dataset(self):
        return self.data_snipper

    def _predict_field_on_points(self, points):
        return np.asarray(points, dtype=np.float64)

    def _to_numpy_cpu(self, data):
        return np.asarray(data)

    def _interpolate_to_test_grid(self, U_train, data_source=None):
        del U_train
        self.interpolation_source = data_source
        return np.zeros((self.data_test["x"].shape[0], 1))

    def _save_final_comparison(self, U_test, final_stage: str):
        self.saved = (np.asarray(U_test), final_stage)


class _FactoryHarness(FixIterationSolverFactoryMixin):
    def __init__(self):
        self.config = object()
        self.data_train = {"name": "train"}
        self.data_snipper = {"name": "snipper"}

    def _get_iteration_dataset(self):
        return self.data_snipper


def test_fixiteration_network_prediction_uses_iteration_dataset():
    harness = _RuntimeHarness()

    prediction = harness._get_network_prediction()

    assert np.array_equal(harness.captured_points, harness.data_snipper["x"])
    assert np.array_equal(prediction, harness.data_snipper["x"])


def test_fixiteration_output_interpolates_from_iteration_dataset():
    harness = _OutputHarness()

    harness._postprocess_results(
        U_final=np.array([[1.0], [2.0]]),
        coeffs_final=np.array([1.0]),
        conv_info={"residual_history": [1.0e-3]},
    )

    assert harness.interpolation_source is harness.data_snipper
    assert harness.saved is not None
    assert harness.saved[1] == "snipper"


def test_fixiteration_factory_builds_fitter_with_iteration_dataset(monkeypatch):
    captured = {}

    class _DummyFitter:
        def __init__(self, config, data=None):
            captured["config"] = config
            captured["data"] = data

    monkeypatch.setattr(factory_module, "FixIterationPDEFitter", _DummyFitter)

    harness = _FactoryHarness()
    fitter = harness._create_fitter(time_scheme=None)

    assert isinstance(fitter, _DummyFitter)
    assert captured["data"] is harness.data_snipper
