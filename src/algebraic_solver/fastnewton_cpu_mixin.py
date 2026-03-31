import numpy as np


class FastNewtonCPUMixin:
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
        """CPU solve implementation."""
        state = self._initialize_state(shape, initial_guess)

        for iteration in range(max_iter):
            f_val, J_raw = build_jacobian_fn(state["x"], A, b, equations, variables)
            f_norm = float(np.linalg.norm(f_val))
            self._update_best_solution(state, f_norm)
            if f_norm < tol:
                break

            J = self._select_jacobian(state, iteration, J_raw)
            try:
                delta = self._compute_cpu_step(J, f_val, state["damping"])
                x_new, new_f_norm = self._run_line_search(
                    state,
                    delta,
                    f_norm,
                    lambda trial_x: build_jacobian_fn(trial_x, A, b, equations, variables)[0],
                )
                self._accept_or_damp(state, x_new, new_f_norm, f_norm)
            except np.linalg.LinAlgError:
                self._apply_fallback_step(
                    state,
                    -J.T @ f_val,
                    f_norm,
                    lambda trial_x: build_jacobian_fn(trial_x, A, b, equations, variables)[0],
                )

            self._print_progress(
                iteration,
                state["total_variables"],
                f_norm,
                state["damping"],
            )

        return self._finalize_solution(state, shape)

    @staticmethod
    def _compute_cpu_step(J, f_val, damping: float):
        JtJ = J.T @ J
        Jtf = J.T @ f_val
        diag = np.diag(JtJ)
        JtJ = JtJ + damping * np.diag(diag)
        L = np.linalg.cholesky(JtJ)
        y = np.linalg.solve(L, -Jtf)
        return np.linalg.solve(L.T, y)
