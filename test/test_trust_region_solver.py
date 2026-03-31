import numpy as np

from src.algebraic_solver.trustregionsolver import TrustRegionSolver


def test_trust_region_subproblem_returns_direct_solution_inside_radius():
    solver = TrustRegionSolver(use_gpu=False, precondition=False)
    B = np.array([[2.0, 0.0], [0.0, 4.0]])
    g = np.array([-2.0, -8.0])

    delta = solver._solve_trust_region_subproblem(B, g, radius=10.0)

    assert np.allclose(delta, np.array([1.0, 2.0]))


def test_trust_region_solver_converges_on_identity_jacobian_system():
    solver = TrustRegionSolver(use_gpu=False, precondition=False)
    target = np.array([[3.0], [4.0]])

    def build_jacobian(x, A, b, equations, variables):
        _ = (A, b, equations, variables)
        f_val = x - target
        J = np.eye(2)
        return f_val, J

    result = solver.solve(
        A=[],
        b=[],
        equations={},
        variables={},
        build_jacobian_fn=build_jacobian,
        shape=(1, 1, 2),
        max_iter=20,
        tol=1e-10,
    )

    assert np.allclose(result.reshape(-1, 1), target, atol=1e-6)
