import numpy as np

from src.algebraic_solver.linear_solver import LinearSolver


class CpuConfig:
    linear_device = "cpu"


def test_linear_solver_respects_cpu_config_override():
    solver = LinearSolver(use_gpu=True, config=CpuConfig())
    assert solver.use_gpu is False


def test_linear_solver_builds_cpu_registry_without_gpu_methods():
    solver = LinearSolver(use_gpu=False)
    assert "direct" in solver.methods
    assert "gpu_direct" not in solver.methods


def test_promote_cpu_method_to_gpu_when_gpu_enabled():
    solver = LinearSolver(use_gpu=False)
    solver.use_gpu = True
    assert solver._promote_cpu_method_to_gpu("direct") == "gpu_direct"
    assert solver._promote_cpu_method_to_gpu("svd") == "gpu_svd"
    assert solver._promote_cpu_method_to_gpu("qr") == "gpu_qr"


def test_resolve_requested_method_uses_auto_selection(monkeypatch):
    solver = LinearSolver(use_gpu=False)
    monkeypatch.setattr(solver, "_auto_select_method", lambda A: "svd")
    A = np.ones((4, 2))
    assert solver._resolve_requested_method("auto", A) == "svd"


def test_get_fallback_method_prefers_direct_for_cpu():
    solver = LinearSolver(use_gpu=False)
    assert solver._get_fallback_method("svd") == "direct"
    assert solver._get_fallback_method("direct") is None


def test_get_fallback_method_routes_failed_gpu_dense_solves_to_cpu_solver(monkeypatch):
    solver = LinearSolver(use_gpu=False)
    solver.use_gpu = True
    monkeypatch.setattr(solver, "_select_cpu_method", lambda A, is_sparse, problem_size: "parallel_svd")
    A = np.ones((8, 4))
    assert solver._get_fallback_method("gpu_lstsq", A) == "parallel_svd"


def test_solve_falls_back_from_gpu_auto_failure_to_cpu_selected_method(monkeypatch):
    solver = LinearSolver(use_gpu=False)
    solver.use_gpu = True
    A = np.ones((8, 4))
    b = np.ones((8, 1))

    def fail(*args, **kwargs):
        raise RuntimeError("gpu unavailable")

    def solve_svd(A, b, **kwargs):
        return np.zeros((A.shape[1], b.shape[1])), {"backend": "cpu"}

    monkeypatch.setattr(solver, "_select_cpu_method", lambda A, is_sparse, problem_size: "svd")
    monkeypatch.setattr(solver, "_auto_select_method", lambda A: "gpu_lstsq")
    solver.methods["gpu_lstsq"] = fail
    solver.methods["svd"] = solve_svd

    x, info = solver.solve(A, b, method="gpu_auto")

    assert x.shape == (4, 1)
    assert info["method"] == "svd"


def test_explicit_gpu_method_does_not_silently_demote_when_gpu_unavailable():
    solver = LinearSolver(use_gpu=False)
    A = np.ones((8, 4))
    b = np.ones((8, 1))

    try:
        solver.solve(A, b, method="gpu_svd")
    except RuntimeError as exc:
        assert "gpu_svd" in str(exc)
    else:
        raise AssertionError("Expected explicit gpu_svd request to fail loudly")


def test_explicit_gpu_method_does_not_fallback_on_runtime_failure(monkeypatch):
    solver = LinearSolver(use_gpu=False)
    solver.use_gpu = True
    A = np.ones((8, 4))
    b = np.ones((8, 1))

    def fail(*args, **kwargs):
        raise RuntimeError("cusolver failed")

    solver.methods["gpu_svd"] = fail

    try:
        solver.solve(A, b, method="gpu_svd")
    except RuntimeError as exc:
        assert "gpu_svd" in str(exc)
        assert "cusolver failed" in str(exc)
    else:
        raise AssertionError("Expected explicit gpu_svd runtime failure to be raised")
