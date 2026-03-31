import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


class PseudoTimeFitterExitMixin:
    def _finalize_max_iterations(
        self,
        state: Dict[str, Any],
        start_time: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        residual = state["residual_prev"] if state["residual_prev"] is not None else float("inf")
        message = f"\nMax iterations reached: Res={residual:.2e} | Time={time.time() - start_time:.1f}s"
        return self._finalize_solver_exit(
            converged=False,
            reason="max_iterations",
            iterations=self.max_iterations,
            residual=residual,
            tau=state["tau"],
            start_time=start_time,
            U=state["U_current"],
            coeffs=state["coeffs_current"],
            message=message,
        )

    def _finalize_solver_exit(
        self,
        *,
        converged: bool,
        reason: str,
        iterations: int,
        residual: float,
        tau: float,
        start_time: float,
        U: np.ndarray,
        coeffs: np.ndarray,
        message: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if message:
            print(message)
        elapsed = time.time() - start_time
        info = self._build_convergence_info(converged, reason, iterations, residual, tau, elapsed)
        return U, coeffs, info

    def _build_convergence_info(
        self,
        converged: bool,
        reason: str,
        iterations: int,
        residual: float,
        tau: float,
        elapsed: float,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build convergence information dictionary."""
        info = {
            "converged": converged,
            "iterations": iterations,
            "final_residual": residual,
            "elapsed_time": elapsed,
            "best_primary_update": getattr(self, "best_primary_update", None),
            "best_iteration": getattr(self, "best_iteration", -1),
            "restart_count": getattr(self, "restart_count", 0),
        }
        if metrics is not None:
            info["final_primary_update"] = metrics["primary_max_abs_update"]
            info["final_primary_relative_update"] = metrics["primary_relative_update"]
            info["final_max_abs_update"] = metrics["max_abs_update"]
        if converged:
            info["final_tau"] = tau
            info["residual_history"] = self.residual_history
            info["dt_history"] = self.dt_history
            info["solution_norm_history"] = self.solution_norm_history
            info["primary_update_history"] = getattr(self, "primary_update_history", [])
            info["primary_relative_update_history"] = getattr(self, "primary_relative_update_history", [])
            info["max_abs_update_history"] = getattr(self, "max_abs_update_history", [])
            info["restart_history"] = getattr(self, "restart_history", [])
        elif reason == "max_iterations":
            info["reason"] = reason
            info["residual_history"] = self.residual_history
            info["dt_history"] = self.dt_history
            info["solution_norm_history"] = self.solution_norm_history
            info["primary_update_history"] = getattr(self, "primary_update_history", [])
            info["primary_relative_update_history"] = getattr(self, "primary_relative_update_history", [])
            info["max_abs_update_history"] = getattr(self, "max_abs_update_history", [])
            info["restart_history"] = getattr(self, "restart_history", [])
        else:
            info["reason"] = reason
            info["restart_history"] = getattr(self, "restart_history", [])
        if extra:
            info.update(extra)
        return info

    def _check_convergence(
        self,
        iteration: int,
        residual: float,
        tau: float,
        start_time: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """Return the convergence message when the residual meets the target."""
        should_check = (iteration + 1) % self.check_interval == 0
        should_check = should_check or residual < self.residual_tol * self.CONVERGENCE_CHECK_FACTOR
        primary_update = None if metrics is None else metrics.get("primary_max_abs_update")
        primary_converged = primary_update is None or primary_update < self.convergence_primary_tol
        if should_check and residual < self.residual_tol and primary_converged:
            elapsed = time.time() - start_time
            primary_msg = "" if primary_update is None else f" | dU_var_max={primary_update:.2e}"
            return (
                f"\nConverged at iter {iteration + 1}: "
                f"Res={residual:.2e}{primary_msg} | tau={tau:.4f} | Time={elapsed:.1f}s"
            )
        return None

    def _check_failure(
        self,
        residual: float,
        stagnation_count: int,
        iteration: int,
    ) -> Optional[Tuple[str, str]]:
        """Check for divergence or stagnation."""
        if residual > self.max_residual:
            msg = f"\nDiverged at iter {iteration + 1}: Res={residual:.2e} > {self.max_residual:.2e}"
            return msg, "divergence"
        if stagnation_count > self.residual_stagnation_steps:
            msg = f"\nStagnated at iter {iteration + 1}: No progress for {stagnation_count} steps"
            return msg, "stagnation"
        return None

    def _print_iteration_progress(
        self,
        iteration: int,
        tau: float,
        dt_used: float,
        dt: float,
        residual: float,
        decrease_rate: float,
        elapsed: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Print iteration progress line."""
        if (iteration + 1) % self.print_interval != 0 and iteration != 0:
            return
        dt_arrow = "↑" if dt > dt_used else ("↓" if dt < dt_used else "")
        metric_msg = ""
        if metrics is not None:
            metric_msg = (
                f" | dU_var_max={metrics['primary_max_abs_update']:.2e}"
                f" | dU_max={metrics['max_abs_update']:.2e}"
            )
        print(
            f"Iter {iteration + 1:5d} | tau={tau:8.4f} | d_tau={dt_used:.2e}{dt_arrow} | "
            f"Res={residual:.2e}(x{decrease_rate:.2f}){metric_msg} | {elapsed:.1f}s"
        )
