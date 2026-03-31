import numpy as np
import scipy.sparse as sp

from src.algebraic_solver.linear_solver import LinearSolver


def test_recommend_method_prefers_parallel_direct_for_tall_dense_matrix():
    A = np.ones((6000, 200))
    method = LinearSolver.recommend_method(A)
    assert method in {"parallel_direct", "qr"}


def test_recommend_method_prefers_lsqr_for_sparse_matrix():
    A = sp.random(1000, 100, density=0.001, format="csr", random_state=0)
    assert LinearSolver.recommend_method(A) == "lsqr"


def test_get_step_summary_aggregates_and_clears_buffer():
    solver = LinearSolver(use_gpu=False)
    solver.current_step_stats = [
        {
            "method": "direct",
            "solve_time": 0.2,
            "residual_norm": 1e-6,
            "relative_residual": 2e-6,
            "condition_number": 10.0,
        },
        {
            "method": "svd",
            "solve_time": 0.3,
            "residual_norm": 5e-6,
            "relative_residual": 4e-6,
            "condition_number": 20.0,
        },
    ]

    summary = solver.get_step_summary(clear_buffer=True)

    assert summary["num_solves"] == 2
    assert np.isclose(summary["total_time"], 0.5)
    assert set(summary["methods_used"]) == {"direct", "svd"}
    assert np.isclose(summary["max_residual"], 5e-6)
    assert np.isclose(summary["max_relative_residual"], 4e-6)
    assert summary["max_condition_number"] == 20.0
    assert solver.current_step_stats == []
