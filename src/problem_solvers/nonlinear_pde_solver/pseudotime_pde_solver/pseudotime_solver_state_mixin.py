import time
from typing import Any, Dict, Tuple

import numpy as np


class PseudoTimeSolverStateMixin:
    def _prepare_iteration_runtime(
        self,
        U: np.ndarray,
        coeffs: np.ndarray,
        tau: float,
    ) -> Dict[str, Any]:
        self.fitter.residual_history = []
        self.fitter.dt_history = []
        self.fitter.solution_norm_history = []
        self.fitter.primary_update_history = []
        self.fitter.primary_relative_update_history = []
        self.fitter.max_abs_update_history = []
        self.fitter.best_primary_update = float("inf")
        self.fitter.best_iteration = -1
        self.fitter.restart_count = 0
        self.fitter.restart_history = []
        self.fitter._coeffs_current = coeffs.copy()
        self._reference_stable_last_primary_update = float("inf")
        self._reference_stable_post_threshold_retrain_done = False

        if hasattr(self.visualizer, "reset_animation"):
            self.visualizer.reset_animation()

        vis_cfg = getattr(self.config, "visualization", None)
        animation_skip = max(1, getattr(vis_cfg, "animation_skip", 1)) if vis_cfg else 1
        return {
            "residual_prev": None,
            "stagnation_count": 0,
            "no_improvement_count": 0,
            "force_train": False,
            "tau_history": [],
            "animation_skip": animation_skip,
            "best_state": self._make_best_state(U, coeffs, tau),
        }

    @staticmethod
    def _make_best_state(
        U: np.ndarray,
        coeffs: np.ndarray,
        tau: float,
        iteration: int = -1,
        primary_metric: float = float("inf"),
    ) -> Dict[str, Any]:
        return {
            "U": U.copy(),
            "coeffs": coeffs.copy(),
            "tau": tau,
            "iteration": iteration,
            "primary_metric": primary_metric,
        }

    def _update_animation_frame(
        self,
        iteration: int,
        residual: float,
        U: np.ndarray,
        animation_skip: int,
    ) -> None:
        if not hasattr(self.visualizer, "update_animation_data"):
            return
        anim_values, anim_data = self._resolve_animation_payload(U)
        try:
            self.visualizer.update_animation_data(
                iteration,
                residual,
                anim_values,
                anim_data,
                animation_skip,
                getattr(self, "reference_solution", None),
            )
        except Exception as exc:
            print(f"  Warning: animation update failed at iter {iteration}: {exc}")

    def _resolve_animation_payload(self, U: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        vis_cfg = getattr(self.config, "visualization", None)
        if not vis_cfg or not getattr(vis_cfg, "should_plot_test", None):
            return U, self.data_train

        if not vis_cfg.should_plot_test():
            return U, self.data_train

        data_test = getattr(self, "data_test", None)
        if not data_test or "x" not in data_test:
            return U, self.data_train

        try:
            U_cpu = self._to_numpy_cpu(U)
            if U_cpu is None:
                return U, self.data_train
            return self._interpolate_to_test_grid(U_cpu), data_test
        except Exception as exc:
            print(f"  Warning: test-grid animation interpolation failed, fallback to train data: {exc}")
            return U, self.data_train

    def _record_best_state(
        self,
        iteration: int,
        U: np.ndarray,
        coeffs: np.ndarray,
        tau_next: float,
        primary_update: float,
        runtime: Dict[str, Any],
    ) -> bool:
        previous_best = self.fitter.best_primary_update
        improved = self.fitter._is_metric_improved(primary_update, previous_best)
        if not improved:
            runtime["no_improvement_count"] += 1
            return False

        if np.isfinite(previous_best):
            print(
                "  Improvement: "
                f"dU_var_max {previous_best:.3e} -> {primary_update:.3e}"
            )
        self.fitter.best_primary_update = primary_update
        self.fitter.best_iteration = iteration
        runtime["best_state"] = self._make_best_state(
            U,
            coeffs,
            tau_next,
            iteration=iteration,
            primary_metric=primary_update,
        )
        runtime["no_improvement_count"] = 0
        return True

    def _handle_plateau_restart(
        self,
        iteration: int,
        primary_update: float,
        dt_used: float,
        runtime: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        if not self.fitter._should_restart_on_plateau(
            primary_update,
            runtime["no_improvement_count"],
        ):
            return False, {}

        best_state = runtime["best_state"]
        restart_dt = self.fitter._get_plateau_restart_dt(dt_used)
        self.fitter.restart_count += 1
        restart_info = {
            "iteration": iteration + 1,
            "best_iteration": best_state["iteration"] + 1 if best_state["iteration"] >= 0 else -1,
            "current_primary_update": primary_update,
            "best_primary_update": best_state["primary_metric"],
            "dt_before": dt_used,
            "dt_after": restart_dt,
            "rollback_tau": best_state["tau"],
        }
        self.fitter.restart_history.append(restart_info)
        print(
            "  Plateau rollback: "
            f"iter={iteration + 1}, best_iter={restart_info['best_iteration']}, "
            f"dU_var_max={primary_update:.3e}, dt {dt_used:.2e}->{restart_dt:.2e}"
        )
        runtime["residual_prev"] = None
        runtime["stagnation_count"] = 0
        runtime["no_improvement_count"] = 0
        runtime["force_train"] = self.fitter.force_retrain_on_plateau
        return True, {
            "U": best_state["U"].copy(),
            "coeffs": best_state["coeffs"].copy(),
            "tau": best_state["tau"],
            "dt": restart_dt,
        }

    def _finalize_iteration_exit(
        self,
        converged: bool,
        reason: str,
        iteration_count: int,
        residual: float,
        tau_next: float,
        start_time: float,
        metrics: Dict[str, Any],
        runtime: Dict[str, Any],
        U: np.ndarray,
        coeffs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        best_state = runtime["best_state"]
        use_best_state = self.fitter.return_best_on_exit and best_state["iteration"] >= 0
        returned_best_state = use_best_state and best_state["iteration"] != (iteration_count - 1)
        returned_U = best_state["U"].copy() if use_best_state else U
        returned_coeffs = best_state["coeffs"].copy() if use_best_state else coeffs
        returned_tau = best_state["tau"] if use_best_state else tau_next
        info = self.fitter._build_convergence_info(
            converged,
            reason,
            iteration_count,
            residual,
            tau_next,
            time.time() - start_time,
            metrics=metrics,
            extra={
                "tau_history": runtime["tau_history"],
                "final_tau": tau_next,
                "returned_tau": returned_tau,
                "returned_best_state": returned_best_state,
                "best_iteration": best_state["iteration"] + 1 if best_state["iteration"] >= 0 else -1,
                "no_improvement_count": runtime["no_improvement_count"],
            },
        )
        return returned_U, returned_coeffs, info
