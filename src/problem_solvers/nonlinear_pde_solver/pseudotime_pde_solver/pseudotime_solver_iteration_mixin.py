import time
from typing import Dict, Tuple

import numpy as np

from .pseudotime_solver_state_mixin import PseudoTimeSolverStateMixin


class PseudoTimeSolverIterationMixin(PseudoTimeSolverStateMixin):
    def _solve_iterations(
        self,
        U,
        dt,
        tau,
        coeffs,
        start_time,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        residual = float("inf")
        metrics = None
        runtime = self._prepare_iteration_runtime(U, coeffs, tau)

        for iteration in range(self.fitter.max_iterations):
            U_prev = U.copy()
            dt_used = dt
            self._train_neural_network_step(
                iteration,
                dt_used,
                U,
                force=runtime["force_train"],
            )
            runtime["force_train"] = False
            U_seg = self.fitter.global_to_segments(U_prev)

            U, _, coeffs = self.fitter.time_step(
                U_prev,
                U_seg,
                dt_used,
                coeffs_n=coeffs,
                current_time=tau,
                step=iteration,
            )
            self.fitter._coeffs_current = coeffs.copy()

            metrics = self.fitter._compute_convergence_metrics(U, U_prev)
            residual = metrics["relative_l2_update"]
            solution_norm = metrics["solution_norm"]
            primary_update = metrics["primary_max_abs_update"]
            self._reference_stable_last_primary_update = primary_update
            self.fitter._update_convergence_history(
                residual,
                dt_used,
                solution_norm,
                metrics=metrics,
            )
            runtime["tau_history"].append(tau)
            tau_next = tau + dt_used

            self._update_animation_frame(
                iteration,
                residual,
                U,
                runtime["animation_skip"],
            )
            improved = self._record_best_state(
                iteration,
                U,
                coeffs,
                tau_next,
                primary_update,
                runtime,
            )

            conv_msg = self.fitter._check_convergence(
                iteration,
                residual,
                tau_next,
                start_time,
                metrics=metrics,
            )
            if conv_msg is not None:
                print(conv_msg)
                return self._finalize_iteration_exit(
                    True,
                    "",
                    iteration + 1,
                    residual,
                    tau_next,
                    start_time,
                    metrics,
                    runtime,
                    U,
                    coeffs,
                )

            restarted, rollback_state = self._handle_plateau_restart(
                iteration,
                primary_update,
                dt_used,
                runtime,
            )
            if restarted:
                U = rollback_state["U"]
                coeffs = rollback_state["coeffs"]
                self.fitter._coeffs_current = coeffs.copy()
                tau = rollback_state["tau"]
                dt = rollback_state["dt"]
                continue

            next_dt, runtime["stagnation_count"] = self.fitter._select_next_timestep(
                dt_used,
                residual,
                runtime["residual_prev"],
                runtime["stagnation_count"],
                improved,
                primary_update,
            )

            failure = self.fitter._check_failure(
                residual,
                runtime["stagnation_count"],
                iteration,
            )
            if failure is not None:
                message, reason = failure
                print(message)
                return self._finalize_iteration_exit(
                    False,
                    reason,
                    iteration + 1,
                    residual,
                    tau_next,
                    start_time,
                    metrics,
                    runtime,
                    U,
                    coeffs,
                )

            elapsed = time.time() - start_time
            decrease_rate = self.fitter._compute_decrease_rate(
                residual,
                runtime["residual_prev"],
            )
            self.fitter._print_iteration_progress(
                iteration,
                tau_next,
                dt_used,
                next_dt,
                residual,
                decrease_rate,
                elapsed,
                metrics=metrics,
            )
            runtime["residual_prev"] = residual
            dt = next_dt
            tau = tau_next

        return self._finalize_iteration_exit(
            False,
            "max_iterations",
            self.fitter.max_iterations,
            residual,
            tau,
            start_time,
            metrics,
            runtime,
            U,
            coeffs,
        )
