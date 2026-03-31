import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .pseudotime_fitter_adaptive_mixin import PseudoTimeFitterAdaptiveMixin
from .pseudotime_fitter_exit_mixin import PseudoTimeFitterExitMixin


class PseudoTimeFitterRuntimeMixin(
    PseudoTimeFitterExitMixin,
    PseudoTimeFitterAdaptiveMixin,
):
    def solve_steady_state(
        self,
        U_init: Optional[np.ndarray] = None,
        coeffs_init: Optional[np.ndarray] = None,
        model: Optional[object] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Solve steady-state PDE using pseudo-time stepping."""
        _ = model
        self._ensure_steady_state_ready()
        state = self._initialize_steady_state_state(U_init, coeffs_init)
        start_time = time.time()

        print("\n=== Pseudo-Time Stepping ===")
        print(f"Target tolerance: {self.residual_tol:.2e} | Max iter: {self.max_iterations}")

        for iteration in range(self.max_iterations):
            step_status = self._attempt_time_step(iteration, state, start_time)
            if step_status["result"] is not None:
                return step_status["result"]
            if step_status["retry"]:
                continue

            U_new, U_seg_new, coeffs_new = step_status["step"]
            residual, solution_norm = self._compute_residual_norm(U_new, state["U_current"])
            tau = state["tau"] + state["dt"]
            dt_used = state["dt"]
            decrease_rate = self._compute_decrease_rate(residual, state["residual_prev"])

            self._update_convergence_history(residual, dt_used, solution_norm)
            self._update_visualizer_frame(iteration, residual, U_new)

            convergence_result = self._finalize_if_converged(
                iteration,
                residual,
                tau,
                start_time,
                U_new,
                coeffs_new,
            )
            if convergence_result is not None:
                return convergence_result

            next_dt, stagnation_count = self._update_adaptive_timestep(
                residual,
                state["residual_prev"],
                state["stagnation_count"],
            )
            failure_result = self._finalize_if_failed(
                iteration,
                residual,
                stagnation_count,
                tau,
                start_time,
                U_new,
                coeffs_new,
            )
            if failure_result is not None:
                return failure_result

            elapsed = time.time() - start_time
            self._print_iteration_progress(
                iteration,
                tau,
                dt_used,
                next_dt,
                residual,
                decrease_rate,
                elapsed,
            )
            self._commit_iteration_state(
                state,
                U_new,
                U_seg_new,
                coeffs_new,
                residual,
                tau,
                next_dt,
                stagnation_count,
            )

        return self._finalize_max_iterations(state, start_time)

    def _ensure_steady_state_ready(self) -> None:
        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")
        if self._segment_start_indices is None:
            self._cache_segment_indices()

    def _initialize_steady_state_state(
        self,
        U_init: Optional[np.ndarray],
        coeffs_init: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        U_current = self._initialize_solution(U_init)
        U_seg_current = self.global_to_segments(U_current)
        coeffs_current = coeffs_init if coeffs_init is not None else self._initialize_coefficients()
        return {
            "dt": self.dt_init,
            "tau": 0.0,
            "U_current": U_current,
            "U_seg_current": U_seg_current,
            "coeffs_current": coeffs_current,
            "residual_prev": None,
            "stagnation_count": 0,
            "consecutive_failures": 0,
        }

    def _initialize_solution(self, U_init: Optional[np.ndarray]) -> np.ndarray:
        if U_init is not None:
            return U_init
        return np.zeros((len(self.data["x"]), self.config.n_eqs))

    def _attempt_time_step(
        self,
        iteration: int,
        state: Dict[str, Any],
        start_time: float,
    ) -> Dict[str, Any]:
        try:
            step = self.time_step(
                state["U_current"],
                state["U_seg_current"],
                state["dt"],
                coeffs_n=state["coeffs_current"],
                current_time=state["tau"],
                step=iteration,
            )
            state["consecutive_failures"] = 0
            return {"retry": False, "result": None, "step": step}
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return self._handle_expected_step_failure(iteration, state, start_time, exc)
        except Exception as exc:  # pragma: no cover - defensive compatibility path
            result = self._finalize_solver_exit(
                converged=False,
                reason="unexpected_error",
                iterations=iteration + 1,
                residual=float("inf"),
                tau=state["tau"],
                start_time=start_time,
                U=state["U_current"],
                coeffs=state["coeffs_current"],
                message=f"\nSolver aborted at iter {iteration}: {type(exc).__name__}: {exc}",
            )
            return {"retry": False, "result": result, "step": None}

    def _handle_expected_step_failure(
        self,
        iteration: int,
        state: Dict[str, Any],
        start_time: float,
        exc: Exception,
    ) -> Dict[str, Any]:
        state["consecutive_failures"] += 1
        failure_count = state["consecutive_failures"]
        print(
            f"  Error at iter {iteration} "
            f"(attempt {failure_count}/{self.MAX_CONSECUTIVE_FAILURES}): "
            f"{type(exc).__name__}: {str(exc)[:50]}..."
        )
        if failure_count >= self.MAX_CONSECUTIVE_FAILURES:
            result = self._finalize_solver_exit(
                converged=False,
                reason="consecutive_failures",
                iterations=iteration + 1,
                residual=float("inf"),
                tau=state["tau"],
                start_time=start_time,
                U=state["U_current"],
                coeffs=state["coeffs_current"],
                message=f"\nSolver failed after {failure_count} consecutive errors",
            )
            return {"retry": False, "result": result, "step": None}

        state["dt"] = max(state["dt"] * self.dt_decrease_factor, self.dt_min)
        return {"retry": True, "result": None, "step": None}

    def _update_visualizer_frame(self, iteration: int, residual: float, U_new: np.ndarray) -> None:
        if self.visualizer is None:
            return
        vis_cfg = getattr(self.config, "visualization", None)
        animation_skip = getattr(vis_cfg, "animation_skip", 1) if vis_cfg else 1
        self.visualizer.update_animation_data(
            iteration + 1,
            residual,
            U_new,
            self.data,
            animation_skip,
            reference_solution=None,
        )

    def _finalize_if_converged(
        self,
        iteration: int,
        residual: float,
        tau: float,
        start_time: float,
        U_new: np.ndarray,
        coeffs_new: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        conv_msg = self._check_convergence(iteration, residual, tau, start_time)
        if conv_msg is None:
            return None
        return self._finalize_solver_exit(
            converged=True,
            reason="",
            iterations=iteration + 1,
            residual=residual,
            tau=tau,
            start_time=start_time,
            U=U_new,
            coeffs=coeffs_new,
            message=conv_msg,
        )

    def _finalize_if_failed(
        self,
        iteration: int,
        residual: float,
        stagnation_count: int,
        tau: float,
        start_time: float,
        U_new: np.ndarray,
        coeffs_new: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        failure = self._check_failure(residual, stagnation_count, iteration)
        if failure is None:
            return None
        message, reason = failure
        return self._finalize_solver_exit(
            converged=False,
            reason=reason,
            iterations=iteration + 1,
            residual=residual,
            tau=tau,
            start_time=start_time,
            U=U_new,
            coeffs=coeffs_new,
            message=message,
        )

    def _commit_iteration_state(
        self,
        state: Dict[str, Any],
        U_new: np.ndarray,
        U_seg_new: np.ndarray,
        coeffs_new: np.ndarray,
        residual: float,
        tau: float,
        dt: float,
        stagnation_count: int,
    ) -> None:
        state["U_current"] = U_new
        state["U_seg_current"] = U_seg_new
        state["coeffs_current"] = coeffs_new
        state["residual_prev"] = residual
        state["tau"] = tau
        state["dt"] = dt
        state["stagnation_count"] = stagnation_count
        self._coeffs_current = coeffs_new
