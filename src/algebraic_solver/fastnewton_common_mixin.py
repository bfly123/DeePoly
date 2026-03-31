import numpy as np


class FastNewtonCommonMixin:
    DEFAULT_DAMPING = 1e-6
    JACOBIAN_REFRESH_FREQUENCY = 2
    LINE_SEARCH_STEPS = 3
    FALLBACK_STEP_SCALE = 0.1
    PRINT_INTERVAL = 5

    def _initialize_state(self, shape, initial_guess):
        ns, n_eqs, dgN = shape
        total_variables = dgN * ns * n_eqs
        if initial_guess is not None:
            x = initial_guess.reshape(-1, 1)
        elif self._last_solution is not None:
            x = self._last_solution
        else:
            x = np.zeros((total_variables, 1))
        return {
            "x": x,
            "total_variables": total_variables,
            "best_x": x.copy(),
            "best_residual": float("inf"),
            "damping": self.DEFAULT_DAMPING,
            "last_J": None,
        }

    def _update_best_solution(self, state, f_norm: float):
        if f_norm < state["best_residual"]:
            state["best_residual"] = f_norm
            state["best_x"] = state["x"].copy()

    def _select_jacobian(self, state, iteration: int, J):
        if iteration % self.JACOBIAN_REFRESH_FREQUENCY == 0 or state["last_J"] is None:
            state["last_J"] = J
            return J
        return state["last_J"]

    def _run_line_search(self, state, delta, f_norm, build_trial_fn):
        alpha = 1.0
        x_current = state["x"]
        x_new = x_current + delta
        new_f_norm = f_norm
        for _ in range(self.LINE_SEARCH_STEPS):
            f_new = build_trial_fn(x_new)
            new_f_norm = float(np.linalg.norm(f_new))
            if new_f_norm < f_norm:
                return x_new, new_f_norm
            alpha *= 0.5
            x_new = x_current + alpha * delta
        return x_new, new_f_norm

    def _accept_or_damp(self, state, x_new, new_f_norm: float, f_norm: float):
        if new_f_norm < f_norm:
            state["x"] = x_new
            state["damping"] = max(self.min_damping, state["damping"] * 0.5)
            return
        state["damping"] = min(self.max_damping, state["damping"] * 1.5)

    def _apply_fallback_step(self, state, delta, f_norm, build_trial_fn):
        x_new = state["x"] + self.FALLBACK_STEP_SCALE * delta
        f_new = build_trial_fn(x_new)
        if np.linalg.norm(f_new) < f_norm:
            state["x"] = x_new
        state["damping"] = min(self.max_damping, state["damping"] * 1.5)

    def _print_progress(self, iteration: int, total_variables: int, f_norm: float, damping: float):
        if iteration % self.PRINT_INTERVAL != 0:
            return
        mean_norm = f_norm / np.sqrt(total_variables)
        print(f"Iteration {iteration + 1}")
        print(f"Mean residual: {mean_norm:.6f}")
        print(f"Damping: {damping:.6e}\n")

    def _finalize_solution(self, state, shape):
        ns, n_eqs, dgN = shape
        self._last_solution = state["best_x"]
        return state["best_x"].reshape(ns, n_eqs, dgN)
