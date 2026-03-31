from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.abstract_class.feature_generator_nn_mixin import FeatureGeneratorNNMixin
from src.abstract_class.spotter_feature_provider import (
    PaddleSpotterFeatureProvider,
    TorchSpotterFeatureProvider,
    create_spotter_feature_provider,
)
from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.problem_solvers.nonlinear_pde_solver.solver import NonlinearPDESolver
from src.spotter_runtime import SpotterOrchestrator


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
        self.output = torch.nn.Linear(3, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.hidden.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -0.5],
                        [0.25, 0.75],
                        [-0.8, 0.3],
                    ],
                    dtype=torch.float64,
                )
            )
            self.output.weight.copy_(torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64))

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        return h, self.output(h)


class DummyProvider:
    backend_name = "dummy"

    def __init__(self):
        self.predict_calls = 0

    def hidden_features(self, x_norm, x_min, x_max):
        _ = (x_min, x_max)
        return np.full((x_norm.shape[0], 2), 3.0, dtype=np.float64)

    def hidden_feature_derivatives(self, x_norm, x_min, x_max, derivative):
        _ = (x_min, x_max)
        value = float(np.sum(derivative) + 1.0)
        return np.full((x_norm.shape[0], 2), value, dtype=np.float64)

    def predict_output(self, points):
        self.predict_calls += 1
        return np.sum(points, axis=1, keepdims=True)


class DummyFitter:
    def __init__(self):
        self.last_source = None

    def fitter_init(self, source):
        self.last_source = source


class DummySolver(NonlinearPDESolver):
    def solve(self, **kwargs):
        raise NotImplementedError

    def _create_config(self, case_dir: str):
        _ = case_dir
        raise NotImplementedError

    def _create_data_generator(self):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError

    def _create_fitter(self, time_scheme):
        _ = time_scheme
        raise NotImplementedError

    def _create_visualizer(self):
        raise NotImplementedError

    def _get_default_time_scheme(self):
        return None

    def _process_reference_solution(self, data, x_ref, mat_file: str):
        _ = (data, x_ref, mat_file)
        raise NotImplementedError


def test_torch_feature_provider_matches_direct_torch_path():
    model = TinyNet()
    provider = TorchSpotterFeatureProvider(model, device="cpu")

    x_norm = np.array([[0.1, 0.2], [0.5, 0.7]], dtype=np.float64)
    x_min = np.array([-1.0, 0.0], dtype=np.float64)
    x_max = np.array([1.0, 2.0], dtype=np.float64)

    direct_hidden = FeatureGeneratorNNMixin.spotter_features(
        x_norm,
        x_min,
        x_max,
        model,
        derivative=[0, 0],
        device="cpu",
    )
    provider_hidden = FeatureGeneratorNNMixin.spotter_features(
        x_norm,
        x_min,
        x_max,
        provider,
        derivative=[0, 0],
        device="cpu",
    )
    np.testing.assert_allclose(provider_hidden, direct_hidden)

    direct_dx = FeatureGeneratorNNMixin.spotter_features(
        x_norm,
        x_min,
        x_max,
        model,
        derivative=[1, 0],
        device="cpu",
    )
    provider_dx = FeatureGeneratorNNMixin.spotter_features(
        x_norm,
        x_min,
        x_max,
        provider,
        derivative=[1, 0],
        device="cpu",
    )
    np.testing.assert_allclose(provider_dx, direct_dx)


def test_paddle_feature_provider_computes_hidden_and_derivatives():
    paddle = pytest.importorskip("paddle")

    def hidden_batch_fn(points):
        x0 = points[:, 0:1]
        x1 = points[:, 1:2]
        return paddle.concat([x0 + 2.0 * x1, x0 * x1], axis=1)

    def output_batch_fn(points):
        return points[:, 0:1] - points[:, 1:2]

    provider = PaddleSpotterFeatureProvider(
        hidden_batch_fn=hidden_batch_fn,
        output_batch_fn=output_batch_fn,
        device="cpu",
    )

    x_norm = np.array([[0.1, 0.2], [0.5, 0.7]], dtype=np.float64)
    x_min = np.zeros(2, dtype=np.float64)
    x_max = np.ones(2, dtype=np.float64)

    hidden = provider.hidden_features(x_norm, x_min, x_max)
    np.testing.assert_allclose(
        hidden,
        np.array([[0.5, 0.02], [1.9, 0.35]], dtype=np.float64),
    )

    dx = provider.hidden_feature_derivatives(x_norm, x_min, x_max, np.array([1, 0]))
    np.testing.assert_allclose(
        dx,
        np.array([[1.0, 0.2], [1.0, 0.7]], dtype=np.float64),
        atol=1e-6,
    )

    dy = provider.hidden_feature_derivatives(x_norm, x_min, x_max, np.array([0, 1]))
    np.testing.assert_allclose(
        dy,
        np.array([[2.0, 0.1], [2.0, 0.5]], dtype=np.float64),
        atol=1e-6,
    )

    prediction = provider.predict_output(np.array([[1.0, 0.25]], dtype=np.float64))
    np.testing.assert_allclose(prediction, np.array([[0.75]], dtype=np.float64), atol=1e-6)


def test_create_spotter_feature_provider_loads_external_factory(tmp_path: Path):
    builder_path = tmp_path / "builder.py"
    builder_path.write_text(
        """
import numpy as np


class BuiltProvider:
    backend_name = "jax"

    def hidden_features(self, x_norm, x_min, x_max):
        _ = (x_min, x_max)
        return np.ones((x_norm.shape[0], 4), dtype=np.float64)

    def hidden_feature_derivatives(self, x_norm, x_min, x_max, derivative):
        _ = (x_min, x_max)
        return np.full((x_norm.shape[0], 4), np.sum(derivative), dtype=np.float64)

    def predict_output(self, points):
        return np.sum(points, axis=1, keepdims=True)


def build_provider(config, model, case_dir, data_train, data_test, provider_class):
    _ = (config, model, case_dir, data_train, data_test, provider_class)
    return BuiltProvider()
""",
        encoding="utf-8",
    )

    config = SimpleNamespace(
        case_dir=str(tmp_path),
        device="cpu",
        spotter_feature_backend="jax",
        spotter_feature={
            "factory_file": "builder.py",
            "factory_function": "build_provider",
        },
    )

    provider = create_spotter_feature_provider(config, TinyNet(), data_train={"x": np.zeros((1, 2))})
    assert provider.backend_name == "jax"
    np.testing.assert_allclose(provider.predict_output(np.array([[1.0, 2.0]])), np.array([[3.0]]))


def test_create_spotter_feature_provider_loads_external_paddle_factory(tmp_path: Path):
    pytest.importorskip("paddle")

    builder_path = tmp_path / "builder_paddle.py"
    builder_path.write_text(
        """
def build_provider(config, model, case_dir, data_train, data_test, provider_class):
    _ = (config, model, case_dir, data_train, data_test)
    import paddle

    def hidden_batch_fn(points):
        return paddle.concat([points[:, :1], 2.0 * points[:, 1:2]], axis=1)

    def output_batch_fn(points):
        return paddle.sum(points, axis=1, keepdim=True)

    return provider_class(
        hidden_batch_fn=hidden_batch_fn,
        output_batch_fn=output_batch_fn,
        device="cpu",
    )
""",
        encoding="utf-8",
    )

    config = SimpleNamespace(
        case_dir=str(tmp_path),
        device="cpu",
        spotter_feature_backend="paddle",
        spotter_feature={
            "factory_file": "builder_paddle.py",
            "factory_function": "build_provider",
        },
    )

    provider = create_spotter_feature_provider(config, TinyNet(), data_train={"x": np.zeros((1, 2))})
    assert provider.backend_name == "paddle"
    np.testing.assert_allclose(provider.predict_output(np.array([[1.0, 2.0]])), np.array([[3.0]]), atol=1e-6)


def test_external_feature_backend_skips_torch_training_and_reinitializes_fitter():
    solver = DummySolver.__new__(DummySolver)
    solver.config = SimpleNamespace(
        device="cpu",
        spotter_feature_backend="jax",
        spotter_feature={},
    )
    solver.model = TinyNet()
    solver.data_train = {"x": np.zeros((2, 2), dtype=np.float64)}
    solver.spotter_feature_provider = DummyProvider()
    solver.fitter = DummyFitter()
    solver._should_train_network = lambda iteration: True

    solver._train_neural_network_step(iter=0, dt=0.1, U_current=np.zeros((2, 1), dtype=np.float64))

    assert solver.fitter.last_source is solver.spotter_feature_provider
    assert solver._last_training_loss is None


def test_predict_current_train_field_prefers_external_provider_output():
    solver = DummySolver.__new__(DummySolver)
    solver.config = SimpleNamespace(device="cpu")
    solver.model = TinyNet()
    solver.data_train = {"x": np.array([[0.2, 0.3], [1.0, -1.0]], dtype=np.float64)}
    solver.spotter_feature_provider = DummyProvider()

    prediction = solver._predict_current_train_field()

    np.testing.assert_allclose(
        prediction,
        np.array([[0.5], [0.0]], dtype=np.float64),
    )
    assert solver.spotter_feature_provider.predict_calls == 1


def test_dynamic_feature_dim_overrides_config_dnn_degree():
    holder = SimpleNamespace(
        config=SimpleNamespace(method="hybrid", device="cpu", DNN_degree=128, x_min=np.zeros((1, 2)), x_max=np.ones((1, 2))),
        data={"x_segments_norm": [np.array([[0.25, 0.5]], dtype=np.float64)]},
        dg=9,
        nn_feature_dim=128,
        dgN=137,
    )

    class FixedDimProvider:
        def __init__(self):
            self.feature_dim = 5

        def get_feature_dim(self, **kwargs):
            _ = kwargs
            return self.feature_dim

    BaseDeepPolyFitter._update_feature_dimensions(holder, FixedDimProvider())

    assert holder.nn_feature_dim == 5
    assert holder.dgN == 14


def test_spotter_orchestrator_parallel_select_chooses_lowest_test_error(tmp_path: Path):
    pytest.importorskip("jax")

    builder_path = tmp_path / "jax_builder.py"
    builder_path.write_text(
        """
import jax.numpy as jnp


def build_provider(config, model, case_dir, data_train, data_test, provider_class, *, offset):
    _ = (config, model, case_dir, data_train, data_test)

    def hidden_fn(point):
        return jnp.asarray([point[0] + offset, point[1] - offset], dtype=jnp.float64)

    def output_fn(point):
        return jnp.asarray([point[0] + point[1] + offset], dtype=jnp.float64)

    return provider_class(hidden_fn=hidden_fn, output_fn=output_fn)
""",
        encoding="utf-8",
    )

    config = SimpleNamespace(
        case_dir=str(tmp_path),
        device="cpu",
        spotter={
            "mode": "parallel_select",
            "selector": {"metric": "test_l2", "max_points": 16},
            "candidates": [
                {
                    "name": "good_jax",
                    "backend": "jax",
                    "runtime": "inprocess",
                    "priority": 1,
                    "feature": {
                        "factory_file": "jax_builder.py",
                        "factory_function": "build_provider",
                        "factory_kwargs": {"offset": 0.0},
                    },
                },
                {
                    "name": "bad_jax",
                    "backend": "jax",
                    "runtime": "inprocess",
                    "priority": 2,
                    "feature": {
                        "factory_file": "jax_builder.py",
                        "factory_function": "build_provider",
                        "factory_kwargs": {"offset": 1.0},
                    },
                },
            ],
        },
    )

    data_test = {
        "x": np.array([[0.0, 0.5], [1.0, -0.5], [0.5, 0.25]], dtype=np.float64),
        "u": np.array([[0.5], [0.5], [0.75]], dtype=np.float64),
    }

    orchestrator = SpotterOrchestrator(
        config=config,
        model=TinyNet(),
        data_train={"x": data_test["x"]},
        data_test=data_test,
        reference_solution=None,
    )
    provider = orchestrator.initialize()

    assert provider.candidate_name == "good_jax"
    assert orchestrator.selection_report is not None
    assert orchestrator.selection_report.selected_name == "good_jax"
    bad_report = next(
        report for report in orchestrator.selection_report.candidates if report.name == "bad_jax"
    )
    good_report = next(
        report for report in orchestrator.selection_report.candidates if report.name == "good_jax"
    )
    assert good_report.score < bad_report.score
    orchestrator.close()


def test_spotter_orchestrator_composite_metric_uses_boundary_residual(tmp_path: Path):
    pytest.importorskip("jax")

    builder_path = tmp_path / "jax_builder_boundary.py"
    builder_path.write_text(
        """
import jax.numpy as jnp


def build_provider(config, model, case_dir, data_train, data_test, provider_class, *, offset):
    _ = (config, model, case_dir, data_train, data_test)

    def hidden_fn(point):
        return jnp.asarray([point[0], point[1]], dtype=jnp.float64)

    def output_fn(point):
        return jnp.asarray([point[0] + point[1] + offset], dtype=jnp.float64)

    return provider_class(hidden_fn=hidden_fn, output_fn=output_fn)
""",
        encoding="utf-8",
    )

    boundary_dict = {
        0: {
            "dirichlet": {
                "x": np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float64),
                "values": np.array([[0.5], [1.0]], dtype=np.float64),
                "weights": np.ones((2, 1), dtype=np.float64),
            },
            "neumann": {"x": np.array([]), "values": np.array([]), "weights": np.array([]), "normals": np.array([])},
            "robin": {"x": np.array([]), "values": np.array([]), "weights": np.array([]), "params": [], "normals": np.array([])},
            "periodic": {"pairs": []},
        }
    }

    config = SimpleNamespace(
        case_dir=str(tmp_path),
        device="cpu",
        spotter={
            "mode": "parallel_select",
            "selector": {
                "metric": "composite",
                "weights": {
                    "relative_l2": 0.0,
                    "boundary_residual": 1.0,
                    "priority": 0.0,
                },
            },
            "candidates": [
                {
                    "name": "good_boundary",
                    "backend": "jax",
                    "runtime": "inprocess",
                    "feature": {
                        "factory_file": "jax_builder_boundary.py",
                        "factory_function": "build_provider",
                        "factory_kwargs": {"offset": 0.0},
                    },
                },
                {
                    "name": "bad_boundary",
                    "backend": "jax",
                    "runtime": "inprocess",
                    "feature": {
                        "factory_file": "jax_builder_boundary.py",
                        "factory_function": "build_provider",
                        "factory_kwargs": {"offset": 0.25},
                    },
                },
            ],
        },
    )

    orchestrator = SpotterOrchestrator(
        config=config,
        model=TinyNet(),
        data_train={"x": boundary_dict[0]["dirichlet"]["x"], "global_boundary_dict": boundary_dict},
        data_test=None,
        reference_solution=None,
    )
    provider = orchestrator.initialize()

    assert provider.candidate_name == "good_boundary"
    selected = next(
        report for report in orchestrator.selection_report.candidates if report.name == "good_boundary"
    )
    rejected = next(
        report for report in orchestrator.selection_report.candidates if report.name == "bad_boundary"
    )
    assert selected.metrics["boundary_residual"] < rejected.metrics["boundary_residual"]
    orchestrator.close()


def test_spotter_orchestrator_subprocess_candidate_round_trip(tmp_path: Path):
    pytest.importorskip("jax")

    builder_path = tmp_path / "jax_builder_subprocess.py"
    builder_path.write_text(
        """
import jax.numpy as jnp


def build_provider(config, model, case_dir, data_train, data_test, provider_class):
    _ = (config, model, case_dir, data_train, data_test)

    def hidden_fn(point):
        return jnp.asarray([point[0] + 1.0, 2.0 * point[1]], dtype=jnp.float64)

    def output_fn(point):
        return jnp.asarray([point[0] - point[1]], dtype=jnp.float64)

    return provider_class(hidden_fn=hidden_fn, output_fn=output_fn)
""",
        encoding="utf-8",
    )

    config = SimpleNamespace(
        case_dir=str(tmp_path),
        device="cpu",
        spotter={
            "mode": "single",
            "selector": {"metric": "priority"},
            "candidates": [
                {
                    "name": "jax_worker",
                    "backend": "jax",
                    "runtime": "subprocess",
                    "python": sys.executable,
                    "feature": {
                        "factory_file": "jax_builder_subprocess.py",
                        "factory_function": "build_provider",
                    },
                }
            ],
        },
    )

    orchestrator = SpotterOrchestrator(
        config=config,
        model=TinyNet(),
        data_train={"x": np.zeros((2, 2), dtype=np.float64)},
        data_test=None,
        reference_solution=None,
    )
    provider = orchestrator.initialize()

    hidden = provider.hidden_features(
        np.array([[0.2, 0.3]], dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.ones(2, dtype=np.float64),
    )
    np.testing.assert_allclose(hidden, np.array([[1.2, 0.6]], dtype=np.float64), atol=1e-6)
    prediction = provider.predict_output(np.array([[1.0, 0.25]], dtype=np.float64))
    np.testing.assert_allclose(prediction, np.array([[0.75]], dtype=np.float64), atol=1e-6)
    orchestrator.close()


def test_solver_reselects_parallel_spotter_after_torch_training_skip():
    solver = DummySolver.__new__(DummySolver)
    solver.config = SimpleNamespace(
        device="cpu",
        spotter_feature_backend="jax",
        spotter_feature={},
    )
    solver.model = TinyNet()
    solver.data_train = {"x": np.zeros((2, 2), dtype=np.float64)}
    solver.spotter_feature_provider = DummyProvider()
    solver.spotter_feature_provider.backend_name = "jax"
    solver.spotter_feature_provider.allow_torch_training = False
    solver.fitter = DummyFitter()

    class DummyOrchestrator:
        def __init__(self, provider):
            self.provider = provider
            self.reselect_calls = 0

        def reselect(self):
            self.reselect_calls += 1
            return self.provider

        def save_report(self, results_dir):
            _ = results_dir

    solver.spotter_orchestrator = DummyOrchestrator(solver.spotter_feature_provider)
    solver._should_train_network = lambda iteration: True

    solver._train_neural_network_step(iter=0, dt=0.1, U_current=np.zeros((2, 1), dtype=np.float64))

    assert solver.spotter_orchestrator.reselect_calls == 1
    assert solver.fitter.last_source is solver.spotter_feature_provider
