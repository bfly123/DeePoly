import os
import time
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


class TimeSolverRuntimeMixin:
    def solve(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """Main solve function"""
        print("=== Starting Time PDE Solving ===")
        start_time = time.time()
        U_final, U_seg_final, model_final, coeffs_final = self.solve_time_evolution()
        total_time = time.time() - start_time
        print(f"Total solving time: {total_time:.2f} seconds")
        self._postprocess_results(U_final, U_seg_final, model_final, coeffs_final)
        return U_final, U_seg_final, model_final, coeffs_final

    def solve_time_evolution(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """Time evolution solving using IMEX-RK(2,2,2) method"""
        it, T, dt, U, U_seg, coeffs = self._initialize_time_evolution()
        self.physics_start_time = time.time()

        while T < self.config.T:
            dt = self._compute_adaptive_timestep(it, T, dt, U)
            self._print_timestep_banner(it, T, dt)
            if hasattr(self.fitter, "solver"):
                self.fitter.solver.reset_step_counter()

            self._train_neural_network_step(it, dt, U_current=U)
            U, U_seg, coeffs = self.fitter.time_step(U, U_seg, dt, coeffs_n=coeffs, step=it, current_time=T)
            self._print_timestep_summary(it, T, U)

            T += dt
            it += 1
            self.visualizer.update_animation_data(
                it,
                T,
                self.data_train,
                self.model,
                coeffs,
                self.fitter,
                solver=self,
                U_direct=U,
                U_seg_direct=U_seg,
            )
            self._monitor_solution(it, U)

        print(f"Time evolution completed. Final time: T = {T:.6f}")
        self.physics_end_time = time.time()
        self.physics_compute_time = self.physics_end_time - self.physics_start_time

        if getattr(self.config, "realtime_visualization", False):
            self.visualizer.finalize_realtime_animation()
        time_history, solution_history = self.visualizer.get_animation_data()
        print(f"Collected {len(time_history)} time steps for animation")
        return U, U_seg, self.model, coeffs

    def _print_timestep_banner(self, it: int, T: float, dt: float) -> None:
        print(f"\n{'=' * 70}")
        print(f"Time Step {it}: T = {T:.6f}, dt = {dt:.6f}")
        print(f"{'=' * 70}")

    def _initialize_time_evolution(self) -> Tuple[int, float, float, np.ndarray, list, np.ndarray]:
        """Initialize time evolution parameters and solution"""
        it = 0
        T = 0.0
        dt = self.config.dt
        U, U_seg = self.data_train["U"], self.data_train["U_seg"]
        coeffs = None
        self.fitter.fitter_init(self._get_spotter_feature_source())
        print(f"Initialized at T = {T:.6f}, ready for time stepping")
        return it, T, dt, U, U_seg, coeffs

    def _compute_adaptive_timestep(self, it: int, T: float, dt: float, U: np.ndarray) -> float:
        """Compute自适应Time step size"""
        _ = dt
        base_dt = self.config.dt
        dt = base_dt
        time_scheme = getattr(self.config, "time_scheme", "imex_rk_222")
        if time_scheme == "onestep_predictor" and it == 0:
            dt *= 0.1
            print(f"  OneStep Predictor: First step dt reduction, dt = {dt:.6f}")
        elif it == 0 and hasattr(self.config, "initial_dt_factor"):
            dt *= self.config.initial_dt_factor

        if T + dt > self.config.T:
            dt = self.config.T - T
        self._previous_U = U.copy()
        return dt

    def _compute_cfl_timestep(self, U: np.ndarray) -> float:
        """Based onCFLConditionComputeTime step size"""
        u_max = np.max(np.abs(U))
        dx_min = np.min(np.diff(self.data_train["x"].flatten()))
        cfl_number = getattr(self.config, "cfl_number", 0.5)
        characteristic_speed = max(u_max, 1e-10)
        return cfl_number * dx_min / characteristic_speed

    def _compute_solution_change_limit(self, U_new: np.ndarray, U_old: np.ndarray, dt: float) -> float:
        """Based onSolutionChange率RestrictionTime step"""
        rel_change = np.linalg.norm(U_new - U_old) / (np.linalg.norm(U_old) + 1e-12)
        max_change_rate = getattr(self.config, "max_solution_change_rate", 0.1)
        if rel_change > max_change_rate:
            dt_new = dt * max_change_rate / rel_change
            return max(dt_new, dt * 0.5)
        return dt

    def get_solver_time_info(self) -> Dict[str, Any]:
        """获取时间格式信息 - TimePDE特定覆盖"""
        info = super().get_solver_time_info()
        if hasattr(self.fitter, "time_scheme") and self.fitter.time_scheme is not None:
            info.update(self.fitter.time_scheme.get_scheme_info())
            info["time_type"] = "physical_time"
        return info

    def estimate_stable_dt(self, U_current: np.ndarray, safety_factor: float = 0.8) -> float:
        """估计稳定时间步 - TimePDE特定方法"""
        if not hasattr(self.fitter, "time_scheme") or self.fitter.time_scheme is None:
            raise RuntimeError("No time scheme initialized")
        return self.fitter.time_scheme.estimate_stable_dt(U_current, safety_factor)

    def print_time_scheme_summary(self):
        """打印时间格式摘要 - TimePDE特定方法"""
        info = self.get_solver_time_info()
        print("=" * 50)
        print("       Time Integration Scheme Summary")
        print("=" * 50)
        print(f"Scheme:       {info.get('method', 'N/A')}")
        print(f"Order:        {info.get('order', 'N/A')}")
        print(f"Stages:       {info.get('stages', 'N/A')}")
        print(f"Stability:    {info.get('stability', 'N/A')}")
        if "operators" in info:
            print("\nOperators:")
            for key, value in info["operators"].items():
                print(f"  - {key}: {value}")
        print("=" * 50)

    def _print_timestep_summary(self, step: int, time_value: float, U: np.ndarray) -> None:
        """打印时间步汇总信息"""
        _ = time_value
        print(f"\n{'-' * 70}")
        print(f"Time Step {step} Summary:")
        print(f"{'-' * 70}")
        self._print_network_summary()
        self._print_linear_solver_summary()
        self._print_solution_statistics(U)
        print(f"{'-' * 70}")

    def _print_network_summary(self) -> None:
        print("  Neural Network:")
        if not hasattr(self, "_last_training_loss") or self._last_training_loss is None:
            print(f"    - Skipped (spotter_skip={getattr(self.config, 'spotter_skip', 1)})")
            return
        loss = self._last_training_loss
        print(f"    - Final Loss: {loss:.6e}")
        if len(self.loss_history) > 1:
            prev_loss = self.loss_history[-2]
            if prev_loss > 0 and loss > 1e-16:
                print(f"    - Loss Reduction: {prev_loss / loss:.2f}x from previous step")
            elif loss < 1e-16:
                print("    - Loss Reduction: Loss near machine epsilon (< 1e-16)")
        elif len(self.loss_history) == 1:
            print(f"    - Initial training (baseline loss: {self.loss_history[0]:.6e})")

    def _print_linear_solver_summary(self) -> None:
        if not hasattr(self.fitter, "solver"):
            return
        solver_stats = self.fitter.solver.get_step_summary(clear_buffer=True)
        if solver_stats.get("num_solves", 0) <= 0:
            return
        print("  Linear Solver:")
        print(f"    - Number of Solves: {solver_stats['num_solves']}")
        print(f"    - Total Time: {solver_stats['total_time']:.4f}s")
        print(f"    - Max Residual (L2): {solver_stats['max_residual']:.6e}")
        print(f"    - Max Relative Residual: {solver_stats['max_relative_residual']:.6e}")
        methods = solver_stats.get("methods_used", [])
        if methods:
            print(f"    - Methods: {', '.join(methods)}")
        if solver_stats.get("max_condition_number") is not None:
            print(f"    - Max Condition Number: {solver_stats['max_condition_number']:.2e}")

    def _print_solution_statistics(self, U: np.ndarray) -> None:
        print("  Solution Statistics:")
        print(f"    - ||U||_2: {np.linalg.norm(U):.6e}")
        print(f"    - max|U|: {np.max(np.abs(U)):.6e}")
        print(f"    - mean(U): {np.mean(U):.6e}")

    def _monitor_solution(self, it: int, U: np.ndarray) -> None:
        """Monitor solution progress"""
        _ = (it, U)
        pass

    def _plot_neural_network_predictions(self, it: int, dt: float) -> None:
        """Plot neural network predictions as scatter plot"""
        try:
            print("  Creating neural network prediction scatter plot...")
            x_test = self.data_test["x"].flatten()
            print(f"  Debug: x_test shape: {x_test.shape}, range: [{np.min(x_test):.3f}, {np.max(x_test):.3f}]")
            u_pred_np, u_true = self._predict_test_values()
            self._save_prediction_scatter_plot(x_test, u_pred_np, u_true, it, dt)
        except Exception as exc:
            print(f"  Warning: Failed to create neural network prediction plot: {exc}")

    def _predict_test_values(self):
        x_tensor = torch.tensor(self.data_test["x"], dtype=torch.float64, device=self.config.device)
        x_tensor.requires_grad_(True)
        self.model.eval()
        _, u_pred = self.model(x_tensor)
        u_pred_abs_max = torch.max(torch.abs(u_pred))
        print(f"  Debug: max absolute value of u_pred: {u_pred_abs_max:.6e}")
        if hasattr(self.model, "out"):
            out_weight_norm = torch.norm(self.model.out.weight).item()
            print(f"  Debug: Output layer weight norm: {out_weight_norm:.6e}")
        u_true = self.data_test["u"].flatten() if "u" in self.data_test else None
        if u_true is not None:
            print(f"  Debug: u_true shape: {u_true.shape}, range: [{np.min(u_true):.3f}, {np.max(u_true):.3f}]")
        return u_pred.detach().cpu().numpy().flatten(), u_true

    def _save_prediction_scatter_plot(self, x_test, u_pred_np, u_true, it: int, dt: float) -> None:
        plt.figure(figsize=(12, 8))
        plt.scatter(x_test, u_pred_np, c="red", alpha=0.7, s=30, label="NN Predictions", marker="o")
        if u_true is not None:
            plt.scatter(x_test, u_true, c="blue", alpha=0.5, s=20, label="True Values", marker=".")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title(f"Neural Network Predictions vs True Values (dt={dt:.6f})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        results_dir = os.path.join(self.config.results_dir)
        os.makedirs(results_dir, exist_ok=True)
        time_history, _ = self.visualizer.get_animation_data()
        filename = (
            f"nn_predictions_it_{it:04d}_t_{time_history[-1]:.6f}_dt_{dt:.6f}.png"
            if len(time_history) > 0
            else f"nn_predictions_it_{it:04d}_dt_{dt:.6f}.png"
        )
        output_path = os.path.join(results_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Neural network prediction plot saved to: {output_path}")
        if u_true is not None:
            mse = np.mean((u_pred_np - u_true) ** 2)
            max_error = np.max(np.abs(u_pred_np - u_true))
            print(f"  MSE between NN and true values: {mse:.6e}")
            print(f"  Max absolute error: {max_error:.6e}")
        print(f"  NN prediction range: [{np.min(u_pred_np):.6e}, {np.max(u_pred_np):.6e}]")
