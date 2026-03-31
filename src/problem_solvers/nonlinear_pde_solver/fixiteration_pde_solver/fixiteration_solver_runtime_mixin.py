import time
from typing import Any, Dict, Tuple

import numpy as np


class FixIterationSolverRuntimeMixin:
    def solve(self, U_init: np.ndarray | None = None) -> Dict[str, Any]:
        U = self._initialize_from_config() if U_init is None else U_init.copy()
        self.fitter.fitter_init(self._get_spotter_feature_source())

        U_final, coeffs_final, conv_info = self._solve_with_fixed_point_iteration(U)
        self._postprocess_results(U_final, coeffs_final, conv_info)
        return {
            "solution": U_final,
            "coefficients": coeffs_final,
            "convergence_info": conv_info,
        }

    def _solve_with_fixed_point_iteration(
        self,
        U_init: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        start_time = time.time()
        if getattr(self, "_skip_initial_network_training", False):
            print("  Skipping initial monolithic Spotter retraining; reusing initialized network weights")
            self._skip_initial_network_training = False
            self.fitter.fitter_init(self._get_spotter_feature_source())
        else:
            self._train_neural_network_step(0, 0.01, U_init)
        U_current = self._get_network_prediction()
        U_current_test = self._predict_field_on_points(self.data_test["x"])
        self._save_network_comparison()

        if not bool(getattr(self.config, "run_snipper", True)):
            print("  Snipper disabled; using bc-PINN Spotter prediction as final solution")
            residual_history = []
            conv_info = self._build_convergence_info(
                True,
                0,
                0.0,
                start_time,
                residual_history,
                reason="spotter_only",
            )
            conv_info["final_stage"] = "spotter"
            conv_info["final_test_solution"] = self._to_numpy_cpu(U_current_test)
            self.fitter.residual_history = residual_history
            return U_current, None, conv_info

        residual_history = []
        coeffs_current = None
        residual = float("inf")
        iteration_data = self._get_iteration_dataset()
        omega = float(getattr(self.fitter, "under_relaxation", 1.0))
        omega = min(max(omega, 0.0), 1.0)

        for iteration in range(self.fitter.max_iterations):
            coeffs_new = self.fitter.fit(U_current=U_current)
            U_picard, _ = self.fitter.construct(
                iteration_data,
                self._get_spotter_feature_source(),
                coeffs_new,
            )
            U_picard_test, _ = self.fitter.construct(
                self.data_test,
                self._get_spotter_feature_source(),
                coeffs_new,
            )
            U_picard = self._to_numpy_cpu(U_picard)
            U_picard_test = self._to_numpy_cpu(U_picard_test)
            if omega < 1.0:
                U_new = (1.0 - omega) * U_current + omega * U_picard
                U_new_test = (1.0 - omega) * U_current_test + omega * U_picard_test
            else:
                U_new = U_picard
                U_new_test = U_picard_test
            residual = self._compute_fixed_point_residual(U_new, U_current)
            residual_history.append(residual)

            if self._should_log_iteration(iteration, residual):
                print(
                    f"Iter {iteration + 1:4d} | Residual: {residual:.6e} | "
                    f"||U||: {np.linalg.norm(U_new):.6e} | omega: {omega:.3f}"
                )

            if residual < self.fitter.convergence_tol:
                self.fitter.residual_history = residual_history
                info = self._build_convergence_info(
                    True,
                    iteration + 1,
                    residual,
                    start_time,
                    residual_history,
                )
                info["final_test_solution"] = self._to_numpy_cpu(U_new_test)
                return U_new, coeffs_new, info

            U_current = U_new.copy()
            U_current_test = U_new_test.copy()
            coeffs_current = coeffs_new.copy()

        self.fitter.residual_history = residual_history
        info = self._build_convergence_info(
            False,
            self.fitter.max_iterations,
            residual,
            start_time,
            residual_history,
            reason="max_iterations",
        )
        info["final_test_solution"] = self._to_numpy_cpu(U_current_test)
        return U_current, coeffs_current, info

    @staticmethod
    def _compute_fixed_point_residual(U_new: np.ndarray, U_current: np.ndarray) -> float:
        return np.linalg.norm(U_new - U_current) / (np.linalg.norm(U_new) + 1e-10)

    def _should_log_iteration(self, iteration: int, residual: float) -> bool:
        return iteration % self.fitter.print_interval == 0 or residual < self.fitter.convergence_tol

    def _build_convergence_info(
        self,
        converged: bool,
        iterations: int,
        final_residual: float,
        start_time: float,
        residual_history: list,
        reason: str | None = None,
    ) -> Dict[str, Any]:
        info = {
            "converged": converged,
            "iterations": iterations,
            "final_residual": final_residual,
            "elapsed_time": time.time() - start_time,
            "residual_history": residual_history,
        }
        if reason is not None:
            info["reason"] = reason
        return info

    def _get_network_prediction(self) -> np.ndarray:
        iteration_data = self._get_iteration_dataset()
        return self._predict_field_on_points(iteration_data["x"])
