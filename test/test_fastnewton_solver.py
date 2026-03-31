import numpy as np

from src.algebraic_solver.fastnewton import FastNewtonSolver


def test_fastnewton_cpu_solve_converges_on_identity_system():
    solver = FastNewtonSolver(use_gpu=False)
    target = np.array([[2.0], [3.0]])

    def build_jacobian(x, A, b, equations, variables):
        _ = (A, b, equations, variables)
        return x - target, np.eye(2)

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


def test_fastnewton_cpu_fallback_handles_singular_jacobian():
    solver = FastNewtonSolver(use_gpu=False)
    target = np.array([[0.1], [0.1]])

    def build_jacobian(x, A, b, equations, variables):
        _ = (A, b, equations, variables)
        residual = x - target
        singular_jacobian = np.zeros((2, 2))
        return residual, singular_jacobian

    result = solver.solve(
        A=[],
        b=[],
        equations={},
        variables={},
        build_jacobian_fn=build_jacobian,
        shape=(1, 1, 2),
        max_iter=2,
        tol=1e-12,
    )

    assert result.shape == (1, 1, 2)
