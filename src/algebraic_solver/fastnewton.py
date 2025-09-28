import numpy as np
import cupy as cp
from typing import List, Dict, Tuple, Optional


class FastNewtonSolver:
    """高效且具有基本Stability保证的非Linear solver"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and cp.is_available()
        self._last_solution = None
        self.min_damping = 1e-12
        self.max_damping = 1e-4

    def solve(
        self,
        A: List,
        b: List,
        equations: Dict,
        variables: Dict,
        build_jacobian_fn,
        shape: Tuple[int, int, int],
        max_iter: int = 200,
        tol: float = 1e-8,
        initial_guess: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """快速但仍具有基本Convergent保证的Solvemethod"""
        if self.use_gpu:
            return self._solve_gpu(
                A,
                b,
                equations,
                variables,
                build_jacobian_fn,
                shape,
                max_iter,
                tol,
                initial_guess,
            )
        else:
            return self._solve_cpu(
                A,
                b,
                equations,
                variables,
                build_jacobian_fn,
                shape,
                max_iter,
                tol,
                initial_guess,
            )

    def _solve_cpu(
        self,
        A,
        b,
        equations,
        variables,
        build_jacobian_fn,
        shape,
        max_iter,
        tol,
        initial_guess,
    ):
        """CPUVersion的SolveImplementation"""
        ns, n_eqs, dgN = shape
        total_variables = dgN * ns * n_eqs

        # 智能Initialize
        if initial_guess is not None:
            x = initial_guess.reshape(-1, 1)
        elif self._last_solution is not None:
            x = self._last_solution
        else:
            x = np.zeros((total_variables, 1))

        # record最佳Solution和InitializeParameter
        best_x = x.copy()
        best_residual = float("inf")
        damping = 1e-6
        update_J_freq = 2
        last_J = None

        for iter in range(max_iter):
            f_val, J = build_jacobian_fn(x, A, b, equations, variables)
            f_norm = float(np.linalg.norm(f_val))
            mean_norm = f_norm / np.sqrt(total_variables)

            if f_norm < best_residual:
                best_residual = f_norm
                best_x = x.copy()

            if f_norm < tol:
                break

            if iter % update_J_freq == 0:
                last_J = J
            else:
                J = last_J

            try:
                JtJ = J.T @ J
                Jtf = J.T @ f_val

                diag = np.diag(JtJ)
                JtJ += damping * np.diag(diag)

                L = np.linalg.cholesky(JtJ)
                y = np.linalg.solve(L, -Jtf)
                delta = np.linalg.solve(L.T, y)
                x_new = x + delta

                # 快速LineSearch
                alpha = 1.0
                for _ in range(3):
                    f_new, _ = build_jacobian_fn(x_new, A, b, equations, variables)
                    new_f_norm = float(np.linalg.norm(f_new))

                    if new_f_norm < f_norm:
                        break

                    alpha *= 0.5
                    x_new = x + alpha * delta

                if new_f_norm < f_norm:
                    x = x_new
                    damping = max(self.min_damping, damping * 0.5)
                else:
                    damping = min(self.max_damping, damping * 1.5)

            except np.linalg.LinAlgError:
                delta = -J.T @ f_val
                x_new = x + 0.1 * delta
                f_new, _ = build_jacobian_fn(x_new, A, b, equations, variables)
                if np.linalg.norm(f_new) < f_norm:
                    x = x_new
                damping = min(self.max_damping, damping * 1.5)

            if iter % 5 == 0:
                print(f"Iteration {iter+1}")
                print(f"Mean residual: {mean_norm:.6f}")
                print(f"Damping: {damping:.6e}\n")

        self._last_solution = best_x
        return best_x.reshape(ns, n_eqs, dgN)

    def _solve_gpu(
        self,
        A,
        b,
        equations,
        variables,
        build_jacobian_fn,
        shape,
        max_iter,
        tol,
        initial_guess,
    ):
        """GPUVersion的SolveImplementation"""
        ns, n_eqs, dgN = shape
        total_variables = dgN * ns * n_eqs

        # 智能Initialize
        if initial_guess is not None:
            x = initial_guess.reshape(-1, 1)
        elif self._last_solution is not None:
            x = self._last_solution
        else:
            x = np.zeros((total_variables, 1))

        # GPUInner存Initialize
        x_gpu = cp.asarray(x)
        buffer = cp.empty_like(x_gpu)

        # record最佳Solution和InitializeParameter
        best_x = x.copy()
        best_residual = float("inf")
        damping = 1e-6
        update_J_freq = 2
        last_J = None

        for iter in range(max_iter):
            f_val, J = build_jacobian_fn(x, A, b, equations, variables)
            f_norm = float(np.linalg.norm(f_val))
            mean_norm = f_norm / np.sqrt(total_variables)

            if f_norm < best_residual:
                best_residual = f_norm
                best_x = x.copy()

            if f_norm < tol:
                break

            if iter % update_J_freq == 0:
                last_J = J
            else:
                J = last_J

            try:
                J_gpu = cp.asarray(J)
                f_val_gpu = cp.asarray(f_val)

                JtJ = J_gpu.T @ J_gpu
                Jtf = J_gpu.T @ f_val_gpu

                diag = cp.diag(JtJ)
                JtJ += damping * cp.diag(diag)

                L = cp.linalg.cholesky(JtJ)
                y = cp.linalg.solve(L, -Jtf)
                delta = cp.linalg.solve(L.T, y)

                buffer = x_gpu + delta
                x_new = cp.asnumpy(buffer)

                # 快速LineSearch
                alpha = 1.0
                for _ in range(3):
                    f_new, _ = build_jacobian_fn(x_new, A, b, equations, variables)
                    new_f_norm = float(np.linalg.norm(f_new))

                    if new_f_norm < f_norm:
                        break

                    alpha *= 0.5
                    delta_cpu = cp.asnumpy(delta)
                    x_new = x + alpha * delta_cpu

                if new_f_norm < f_norm:
                    x = x_new
                    x_gpu = cp.asarray(x)
                    damping = max(self.min_damping, damping * 0.5)
                else:
                    damping = min(self.max_damping, damping * 1.5)

            except cp.linalg.LinAlgError:
                delta = -cp.asnumpy(J_gpu.T @ f_val_gpu)
                x_new = x + 0.1 * delta
                f_new, _ = build_jacobian_fn(x_new, A, b, equations, variables)
                if np.linalg.norm(f_new) < f_norm:
                    x = x_new
                    x_gpu = cp.asarray(x)
                damping = min(self.max_damping, damping * 1.5)

            if iter % 5 == 0:
                print(f"Iteration {iter+1}")
                print(f"Mean residual: {mean_norm:.6f}")
                print(f"Damping: {damping:.6e}\n")

        self._last_solution = best_x
        return best_x.reshape(ns, n_eqs, dgN)
