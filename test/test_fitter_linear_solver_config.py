from types import SimpleNamespace

import numpy as np

import src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.core.fitter as pseudotime_fitter_module
from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.core.fitter import (
    PseudoTimePDEFitter,
)


class FakeLinearSolver:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.solve_calls = []

    def solve(self, A, b, method="auto", **kwargs):
        self.solve_calls.append({"A": A, "b": b, "method": method, "kwargs": kwargs})
        return b, {"method": method}


def test_pseudotime_fitter_passes_config_into_linear_solver(monkeypatch):
    captured = {}

    def fake_solver_factory(**kwargs):
        captured.update(kwargs)
        return FakeLinearSolver(**kwargs)

    monkeypatch.setattr(pseudotime_fitter_module, "LinearSolver", fake_solver_factory)

    config = SimpleNamespace(
        n_dim=2,
        n_segments=[1, 1],
        poly_degree=[1, 1],
        linear_device="cpu",
        method="sniper",
        n_eqs=1,
        max_iterations=1,
        residual_tol=1e-6,
        dt_init=0.1,
        dt_min=0.01,
        dt_max=0.1,
        adaptive_dt=False,
        dt_increase_factor=1.2,
        dt_decrease_factor=0.5,
        check_interval=1,
        print_interval=1,
        save_interval=1,
        max_residual=1.0,
        residual_stagnation_steps=1,
        operator_parse={"max_derivative_orders": [[1, 1]], "all_derivatives": {}, "derivatives": [[0, 0]], "operator_terms": {}},
    )

    fitter = PseudoTimePDEFitter(config, data={"x_segments_norm": [None]})

    assert isinstance(fitter.solver, FakeLinearSolver)
    assert captured["config"] is config
    assert captured["use_gpu"] is True


def test_base_fitter_core_uses_configured_linear_method():
    fake_solver = FakeLinearSolver()

    class DummyFitter:
        _solve_system = BaseDeepPolyFitter._solve_system
        _get_linear_solver_kwargs = BaseDeepPolyFitter._get_linear_solver_kwargs

    fitter = DummyFitter()
    fitter.ns = 1
    fitter.dgN = 2
    fitter.config = SimpleNamespace(n_eqs=3, linear_method="lsqr")
    fitter.solver = fake_solver

    def fake_build_jacobian(**kwargs):
        del kwargs
        return (
            np.eye(6, dtype=float),
            np.arange(6, dtype=float).reshape(6, 1),
        )

    fitter._build_jacobian = fake_build_jacobian

    coeffs = fitter._solve_system()

    assert coeffs.shape == (1, 3, 2)
    assert fake_solver.solve_calls[0]["method"] == "lsqr"


def test_base_fitter_core_passes_svd_and_ridge_solver_kwargs():
    class DummyFitter:
        _solve_system = BaseDeepPolyFitter._solve_system
        _get_linear_solver_kwargs = BaseDeepPolyFitter._get_linear_solver_kwargs

    def fake_build_jacobian(**kwargs):
        del kwargs
        return (
            np.eye(2, dtype=float),
            np.arange(2, dtype=float).reshape(2, 1),
        )

    fitter = DummyFitter()
    fitter.ns = 1
    fitter.dgN = 2
    fitter.solver = FakeLinearSolver()
    fitter._build_jacobian = fake_build_jacobian

    fitter.config = SimpleNamespace(
        n_eqs=1,
        linear_method="gpu_svd",
        linear_rcond=1e-8,
        linear_alpha=None,
    )
    fitter._solve_system()
    first_call = fitter.solver.solve_calls[-1]
    assert first_call["method"] == "gpu_svd"
    assert first_call["kwargs"]["rcond"] == 1e-8
    assert "alpha" not in first_call["kwargs"]

    fitter.config = SimpleNamespace(
        n_eqs=1,
        linear_method="parallel_ridge",
        linear_rcond=None,
        linear_alpha=1e-10,
    )
    fitter._solve_system()
    second_call = fitter.solver.solve_calls[-1]
    assert second_call["method"] == "parallel_ridge"
    assert second_call["kwargs"]["alpha"] == 1e-10
    assert "rcond" not in second_call["kwargs"]
