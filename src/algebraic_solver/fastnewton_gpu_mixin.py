import numpy as np

from .linear_solver_backend import CUPY_AVAILABLE, cp


class FastNewtonGPUMixin:
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
        """GPU solve implementation."""
        if not CUPY_AVAILABLE or cp is None:
            raise RuntimeError("CuPy is required for GPU execution in FastNewtonSolver")

        state = self._initialize_state(shape, initial_guess)
        x_gpu = cp.asarray(state["x"])
        buffer = cp.empty_like(x_gpu)

        for iteration in range(max_iter):
            f_val, J_raw = build_jacobian_fn(state["x"], A, b, equations, variables)
            f_norm = float(np.linalg.norm(f_val))
            self._update_best_solution(state, f_norm)
            if f_norm < tol:
                break

            J = self._select_jacobian(state, iteration, J_raw)
            try:
                delta, x_new = self._compute_gpu_step(J, f_val, state["damping"], x_gpu, buffer)
                x_new, new_f_norm = self._run_line_search(
                    state,
                    delta,
                    f_norm,
                    lambda trial_x: build_jacobian_fn(trial_x, A, b, equations, variables)[0],
                )
                self._accept_or_damp(state, x_new, new_f_norm, f_norm)
                x_gpu = cp.asarray(state["x"])
            except cp.linalg.LinAlgError:
                J_gpu = cp.asarray(J)
                f_val_gpu = cp.asarray(f_val)
                delta = -cp.asnumpy(J_gpu.T @ f_val_gpu)
                self._apply_fallback_step(
                    state,
                    delta,
                    f_norm,
                    lambda trial_x: build_jacobian_fn(trial_x, A, b, equations, variables)[0],
                )
                x_gpu = cp.asarray(state["x"])

            self._print_progress(
                iteration,
                state["total_variables"],
                f_norm,
                state["damping"],
            )

        return self._finalize_solution(state, shape)

    @staticmethod
    def _compute_gpu_step(J, f_val, damping, x_gpu, buffer):
        J_gpu = cp.asarray(J)
        f_val_gpu = cp.asarray(f_val)
        JtJ = J_gpu.T @ J_gpu
        Jtf = J_gpu.T @ f_val_gpu
        diag = cp.diag(JtJ)
        JtJ = JtJ + damping * cp.diag(diag)
        L = cp.linalg.cholesky(JtJ)
        y = cp.linalg.solve(L, -Jtf)
        delta_gpu = cp.linalg.solve(L.T, y)
        buffer = x_gpu + delta_gpu
        delta = cp.asnumpy(delta_gpu)
        x_new = cp.asnumpy(buffer)
        return delta, x_new
