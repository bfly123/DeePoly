"""
General PDE Fitter with Pseudo-Time Stepping

Extends TimePDEFitter to solve steady-state PDEs using pseudo-time stepping:
    ∂U/∂τ + R(U) = 0 → U as τ → ∞
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time

from src.problem_solvers.time_pde_solver.core.fitter import TimePDEFitter
from src.problem_solvers.time_pde_solver.time_schemes import create_time_scheme


class GeneralPDEFitter(TimePDEFitter):
    """General PDE Fitter with pseudo-time stepping for steady-state problems

    Features:
    - Pseudo-time stepping iteration
    - Automatic convergence monitoring
    - Adaptive pseudo-time step control
    - Residual-based stopping criteria
    """

    def __init__(self, config, data: Dict = None, time_scheme: str = "implicit_euler", visualizer=None):
        super().__init__(config, data=data, time_scheme=time_scheme)

        # Pseudo-time stepping parameters
        self.max_iterations = config.max_iterations
        self.convergence_tol = config.convergence_tol
        self.residual_tol = config.residual_tol

        # Time step control
        self.dt_init = config.dt_init
        self.dt_min = config.dt_min
        self.dt_max = config.dt_max
        self.adaptive_dt = config.adaptive_dt
        self.dt_increase_factor = config.dt_increase_factor
        self.dt_decrease_factor = config.dt_decrease_factor

        # Monitoring
        self.check_interval = config.check_interval
        self.print_interval = config.print_interval
        self.save_interval = config.save_interval
        self.max_residual = config.max_residual
        self.residual_stagnation_steps = config.residual_stagnation_steps

        # Convergence history
        self.residual_history = []
        self.dt_history = []
        self.solution_norm_history = []

        # Current coefficients (for compute_residual)
        self._coeffs_current = None

        # Visualizer
        self.visualizer = visualizer

    def solve_steady_state(
        self,
        U_init: Optional[np.ndarray] = None,
        coeffs_init: Optional[np.ndarray] = None,
        model: Optional[object] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Solve steady-state PDE using pseudo-time stepping

        Args:
            U_init: Initial guess for solution (optional, defaults to zero)
            coeffs_init: Initial guess for coefficients (optional)
            model: Neural network model for training at each step (optional)

        Returns:
            Tuple: (final solution, final coefficients, convergence info)
        """
        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        # Store model reference for training
        self.model = model

        # Initialize solution
        if U_init is None:
            U_init = np.zeros((len(self.data["x"]), self.config.n_eqs))

        # Convert to segment format
        U_seg = self.global_to_segments(U_init)

        # Initialize coefficients if needed
        if coeffs_init is None:
            coeffs_init = self._initialize_coefficients()

        # Initialize pseudo-time stepping
        dt = self.dt_init
        tau = 0.0
        U_current = U_init.copy()
        U_seg_current = [seg.copy() for seg in U_seg]
        coeffs_current = coeffs_init.copy()

        residual_prev = None
        stagnation_count = 0

        print("=" * 70)
        print("Starting Pseudo-Time Stepping Iteration")
        print("=" * 70)
        print(f"Max iterations: {self.max_iterations}")
        print(f"Convergence tolerance: {self.convergence_tol:.2e}")
        print(f"Residual tolerance: {self.residual_tol:.2e}")
        print(f"Initial dt: {dt:.6f}")
        print("=" * 70)

        start_time = time.time()

        # Pseudo-time stepping loop
        for iteration in range(self.max_iterations):
            # Train neural network at current pseudo-time step (similar to time_pde_solver)
            if self.model is not None:
                self._train_neural_network_step(iteration, dt, U_current)

            # Execute one time step
            try:
                U_new, U_seg_new, coeffs_new = self.time_step(
                    U_current, U_seg_current, dt,
                    coeffs_n=coeffs_current,
                    current_time=tau,
                    step=iteration
                )
            except Exception as e:
                print(f"\nError at iteration {iteration}: {e}")
                print("Reducing time step and retrying...")
                dt *= self.dt_decrease_factor
                dt = max(dt, self.dt_min)
                continue

            # Compute residual: ||U_new - U_current|| / max(||U_new||, ||U_current||, 1.0)
            delta_U = U_new - U_current
            u_norm = np.linalg.norm(U_new)
            u_current_norm = np.linalg.norm(U_current)
            residual = np.linalg.norm(delta_U) / max(u_norm, u_current_norm, 1.0)
            solution_norm = u_norm

            # Store history
            self.residual_history.append(residual)
            self.dt_history.append(dt)
            self.solution_norm_history.append(solution_norm)

            # Update pseudo-time BEFORE printing (so "Iter 1" shows tau after completing step 1)
            tau += dt

            # Update animation data
            if self.visualizer is not None:
                animation_skip = getattr(self.config, 'animation_skip', 1)
                self.visualizer.update_animation_data(
                    iteration + 1, residual, U_new, self.data, animation_skip
                )

            # Print progress (now shows updated tau)
            if (iteration + 1) % self.print_interval == 0 or iteration == 0:
                elapsed = time.time() - start_time
                print(f"Iter {iteration+1:6d} | τ={tau:10.6f} | dt={dt:.6e} | "
                      f"Residual={residual:.6e} | ||U||={solution_norm:.6e} | "
                      f"Time={elapsed:.2f}s")

            # Check convergence
            if (iteration + 1) % self.check_interval == 0:
                if residual < self.convergence_tol:
                    elapsed = time.time() - start_time
                    print("\n" + "=" * 70)
                    print(f"✓ Converged at iteration {iteration+1}!")
                    print(f"  Final residual: {residual:.6e}")
                    print(f"  Final τ: {tau:.6f}")
                    print(f"  Total time: {elapsed:.2f}s")
                    print("=" * 70)

                    convergence_info = {
                        "converged": True,
                        "iterations": iteration + 1,
                        "final_residual": residual,
                        "final_tau": tau,
                        "elapsed_time": elapsed,
                        "residual_history": self.residual_history,
                        "dt_history": self.dt_history,
                        "solution_norm_history": self.solution_norm_history
                    }
                    return U_new, coeffs_new, convergence_info

            # Adaptive time step control
            if self.adaptive_dt and residual_prev is not None:
                if residual < residual_prev * 0.8:
                    # Good progress, increase time step
                    dt = min(dt * self.dt_increase_factor, self.dt_max)
                    stagnation_count = 0
                elif residual > residual_prev * 1.2:
                    # Poor progress or diverging, decrease time step
                    dt = max(dt * self.dt_decrease_factor, self.dt_min)
                    stagnation_count += 1
                else:
                    # Maintain current time step
                    stagnation_count += 1

            # Check for divergence
            if residual > self.max_residual:
                print(f"\n✗ Divergence detected at iteration {iteration+1}!")
                print(f"  Residual: {residual:.6e} > max_residual: {self.max_residual:.6e}")
                convergence_info = {
                    "converged": False,
                    "reason": "divergence",
                    "iterations": iteration + 1,
                    "final_residual": residual
                }
                return U_new, coeffs_new, convergence_info

            # Check for stagnation
            if stagnation_count > self.residual_stagnation_steps:
                print(f"\n✗ Stagnation detected at iteration {iteration+1}!")
                print(f"  No progress for {stagnation_count} steps")
                convergence_info = {
                    "converged": False,
                    "reason": "stagnation",
                    "iterations": iteration + 1,
                    "final_residual": residual
                }
                return U_new, coeffs_new, convergence_info

            # Update for next iteration
            U_current = U_new.copy()
            U_seg_current = [seg.copy() for seg in U_seg_new]
            coeffs_current = coeffs_new.copy()
            self._coeffs_current = coeffs_current  # Update for compute_residual
            residual_prev = residual
            # tau already updated above (before printing)

        # Maximum iterations reached
        elapsed = time.time() - start_time
        print(f"\n✗ Maximum iterations ({self.max_iterations}) reached without convergence")
        print(f"  Final residual: {residual:.6e}")
        print(f"  Total time: {elapsed:.2f}s")

        convergence_info = {
            "converged": False,
            "reason": "max_iterations",
            "iterations": self.max_iterations,
            "final_residual": residual,
            "elapsed_time": elapsed,
            "residual_history": self.residual_history,
            "dt_history": self.dt_history,
            "solution_norm_history": self.solution_norm_history
        }

        return U_current, coeffs_current, convergence_info

    def _initialize_coefficients(self) -> np.ndarray:
        """Initialize coefficient array with zeros or small random values"""
        coeffs = np.zeros((self.ns, self.config.n_eqs, self.dgN))
        # Add small random perturbation to avoid exact symmetry
        coeffs += np.random.randn(*coeffs.shape) * 1e-8
        return coeffs

    def compute_residual(self, U: np.ndarray) -> np.ndarray:
        """Compute residual R(U) = L1(U) + L2(U)·F(U) + N(U) - S

        Args:
            U: Solution array (n_points, n_eqs)

        Returns:
            Residual array (n_points, n_eqs)
        """
        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        # 需要当前系数才能计算L1和L2项
        if not hasattr(self, '_coeffs_current') or self._coeffs_current is None:
            raise RuntimeError("Coefficients not available. Cannot compute L1/L2 terms.")

        U_seg = self.global_to_segments(U)
        residual = np.zeros_like(U)

        for seg_idx in range(self.ns):
            U_seg_i = U_seg[seg_idx]
            features = self._features[seg_idx][0]
            coeffs_seg = self._coeffs_current[seg_idx]

            # Get operators
            L1_ops = self._linear_operators[seg_idx].get("L1", None)
            L2_ops = self._linear_operators[seg_idx].get("L2", None)

            # Compute F and N
            F_val = self.F_func(features, U_seg_i)
            N_val = self.N_func(features, U_seg_i)

            # Get source term S
            S_val = self._get_source_term(seg_idx)

            # Compute residual for this segment
            residual_seg = np.zeros_like(U_seg_i)

            for eq_idx in range(self.config.n_eqs):
                # L1(U) term: 算子矩阵 @ 系数
                if L1_ops is not None and eq_idx < len(L1_ops):
                    residual_seg[:, eq_idx] += L1_ops[eq_idx] @ coeffs_seg[eq_idx]

                # L2(U)·F(U) term
                if L2_ops is not None and eq_idx < len(L2_ops):
                    L2_U = L2_ops[eq_idx] @ coeffs_seg[eq_idx]
                    residual_seg[:, eq_idx] += L2_U * F_val[:, eq_idx]

                # N(U) term
                residual_seg[:, eq_idx] += N_val[eq_idx]

                # -S term
                residual_seg[:, eq_idx] -= S_val[:, eq_idx]

            # Map back to global indices
            start_idx, end_idx = self._get_segment_global_indices(seg_idx)
            residual[start_idx:end_idx] = residual_seg

        return residual

    def _get_source_term(self, segment_idx: int) -> np.ndarray:
        """获取段的源项

        Args:
            segment_idx: 段索引

        Returns:
            S: 源项数组 (n_points, n_eqs)
        """
        if "source_segments" in self.data:
            return self.data["source_segments"][segment_idx]

        # 默认零源项
        n_points = len(self.data["x_segments"][segment_idx])
        return np.zeros((n_points, self.config.n_eqs))

    def _get_segment_global_indices(self, segment_idx: int) -> Tuple[int, int]:
        """Get global array indices for a segment"""
        points_per_segment = [len(seg) for seg in self.data["x_segments_norm"]]
        start_idx = sum(points_per_segment[:segment_idx])
        end_idx = start_idx + points_per_segment[segment_idx]
        return start_idx, end_idx

    def _train_neural_network_step(self, it: int, dt: float, U_current: np.ndarray = None) -> None:
        """Neural network training at current pseudo-time step (参考 time_pde_solver)"""

        # 检查是否需要跳过神经网络训练
        spotter_skip = getattr(self.config, 'spotter_skip', 1)  # 默认每步都训练
        skip_training = (it % spotter_skip != 0) and (it > 0)  # 第一步总是训练

        if skip_training:
            print(f"  Skipping neural network training (spotter_skip={spotter_skip}, it={it})")
            # 仍然需要重新初始化fitter，但使用现有模型参数
            self.fitter_init(self.model)
            return

        print(f"  Training neural network (it={it}, inheriting from previous parameters)...")

        # 确保 model.config 有必要的属性
        fitter_training = getattr(self.config, 'fitter_training', {})

        if not hasattr(self.model.config, 'max_retries'):
            self.model.config.max_retries = fitter_training.get('max_retries', 3)

        if not hasattr(self.model.config, 'learning_rate'):
            self.model.config.learning_rate = fitter_training.get('learning_rate', 0.001)

        if not hasattr(self.model.config, 'n_epoch'):
            self.model.config.n_epoch = fitter_training.get('epochs', 100)

        if not hasattr(self.model.config, 'epochs_adam'):
            self.model.config.epochs_adam = fitter_training.get('epochs', 100)

        if not hasattr(self.model.config, 'epochs_lbfgs'):
            self.model.config.epochs_lbfgs = fitter_training.get('epochs_lbfgs', 0)

        if not hasattr(self.model.config, 'DNNtol'):
            self.model.config.DNNtol = fitter_training.get('DNNtol', 1e-6)

        if not hasattr(self.model.config, 'dt'):
            self.model.config.dt = dt

        # 不重新初始化模型参数，保持从上次训练继承
        data_GPU = self.model.prepare_gpu_data(self.data, U_current)

        # 从上一次的模型状态开始训练（不重置参数）
        self.model.train()  # 设置为训练模式

        # 执行训练，从上次的权重继承
        self.model.train_net(self.data, self.model, data_GPU, dt=dt)

        # 计算最终Loss
        self.model.eval()
        final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
        print(f"  Neural network training completed, loss: {final_loss:.8e}")

        # 重新初始化fitter以使用更新后的模型
        self.fitter_init(self.model)