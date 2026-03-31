import numpy as np

from .linear_solver_backend import cp


class TrustRegionRuntimeMixin:
    ETA_LOW = 0.1
    ETA_HIGH = 0.9
    INITIAL_RADIUS = 2.0
    MAX_RADIUS = 20.0
    MIN_RADIUS = 1e-6
    ACCEPT_THRESHOLD = 0.05

    def solve(
        self,
        A,
        b,
        equations,
        variables,
        build_jacobian_fn,
        shape,
        max_iter: int = 50,
        tol: float = 1e-8,
        initial_guess=None,
    ) -> np.ndarray:
        """Solve the nonlinear system with an adaptive trust-region method."""
        ns, n_eqs, dgN = shape
        state = self._initialize_solver_state(shape, initial_guess)
        workspace = self._initialize_workspace(state["x"])
        best_x = state["x"].copy()
        best_residual = float("inf")

        for iteration in range(max_iter):
            f_val, J = build_jacobian_fn(state["x"], A, b, equations, variables)
            f_norm = float(np.linalg.norm(f_val))
            best_x, best_residual = self._update_best_solution(
                state["x"],
                f_norm,
                best_x,
                best_residual,
            )
            if f_norm < tol:
                break

            scaled_system = self._build_scaled_system(J, f_val)
            delta = self._compute_trust_region_step(scaled_system, state["radius"], workspace)
            delta = self._restore_scaled_step(delta, scaled_system["scale_inverse"])
            step_norm = float(np.linalg.norm(delta))
            delta = self._clip_step_to_radius(delta, step_norm, state["radius"])
            trial_info = self._evaluate_trial_step(
                state["x"],
                delta,
                A,
                b,
                equations,
                variables,
                build_jacobian_fn,
                f_val,
                J,
            )
            state["radius"] = self._update_radius(state["radius"], step_norm, trial_info["rho"])
            if trial_info["rho"] > self.ACCEPT_THRESHOLD:
                state["x"] = state["x"] + delta
            self._print_iteration_diagnostics(iteration, state["radius"], step_norm, trial_info["rho"], f_val)
            if step_norm < tol * (1 + np.linalg.norm(state["x"])):
                break

        self._last_step = best_x
        return best_x.reshape(ns, n_eqs, dgN)

    def _initialize_solver_state(self, shape, initial_guess=None):
        ns, n_eqs, dgN = shape
        if initial_guess is not None:
            x = initial_guess.reshape(-1, 1)
        elif self._last_step is not None:
            x = self._last_step
        else:
            x = np.zeros((dgN * ns * n_eqs, 1))
        return {"x": x, "radius": self.INITIAL_RADIUS}

    def _initialize_workspace(self, x):
        if not self.use_gpu:
            return None
        return {"x": cp.asarray(x)}

    def _update_best_solution(self, x, f_norm, best_x, best_residual):
        if f_norm < best_residual:
            return x.copy(), f_norm
        return best_x, best_residual

    def _build_scaled_system(self, J, f_val):
        gradient = J.T @ f_val
        hessian = J.T @ J
        if not self.precondition:
            return {
                "B_scaled": hessian,
                "g_scaled": gradient,
                "scale_inverse": None,
            }

        scale = np.diag(np.sqrt(np.diag(hessian) + 1e-12))
        scale_inverse = np.diag(1.0 / np.diag(scale))
        return {
            "B_scaled": scale_inverse @ hessian @ scale_inverse,
            "g_scaled": scale_inverse @ gradient,
            "scale_inverse": scale_inverse,
        }

    def _compute_trust_region_step(self, scaled_system, radius, workspace):
        gradient = scaled_system["g_scaled"].flatten()
        if self.use_gpu:
            return self._solve_trust_region_subproblem_gpu(
                scaled_system["B_scaled"],
                gradient,
                radius,
                workspace,
            )
        return self._solve_trust_region_subproblem(
            scaled_system["B_scaled"],
            gradient,
            radius,
        )

    def _restore_scaled_step(self, delta, scale_inverse):
        delta = delta.reshape(-1, 1)
        if scale_inverse is None:
            return delta
        return scale_inverse @ delta

    def _clip_step_to_radius(self, delta, step_norm, radius):
        if step_norm <= radius:
            return delta
        return delta * (radius / step_norm)

    def _evaluate_trial_step(
        self,
        x,
        delta,
        A,
        b,
        equations,
        variables,
        build_jacobian_fn,
        f_val,
        J,
    ):
        new_f_val, _ = build_jacobian_fn(x + delta, A, b, equations, variables)
        actual_reduction = float(np.linalg.norm(f_val) - np.linalg.norm(new_f_val))
        predicted_reduction = float(np.linalg.norm(f_val) - np.linalg.norm(f_val + J @ delta))
        rho = actual_reduction / (predicted_reduction + 1e-10)
        return {"rho": rho, "new_f_val": new_f_val}

    def _update_radius(self, radius, step_norm, rho):
        if rho < self.ETA_LOW:
            return max(0.25 * radius, self.MIN_RADIUS)
        if rho > self.ETA_HIGH and step_norm >= 0.95 * radius:
            return min(2.0 * radius, self.MAX_RADIUS)
        if self.ETA_LOW <= rho <= self.ETA_HIGH:
            return max(0.5 * radius, self.MIN_RADIUS)
        return radius

    def _print_iteration_diagnostics(self, iteration, radius, step_norm, rho, f_val):
        print(f"Iteration {iteration + 1}")
        print(f"Trust radius: {radius:.6f}")
        print(f"Step size: {step_norm:.6f}")
        print(f"Improvement ratio: {rho:.6f}")
        print(f"Residual: {np.mean(np.abs(f_val)):.6f}\n")
