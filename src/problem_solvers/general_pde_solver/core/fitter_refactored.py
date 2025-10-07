"""
General PDE Fitter with Pseudo-Time Stepping (Refactored)

参考 time_pde_solver 的结构，改进代码组织和可读性
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time

from src.problem_solvers.time_pde_solver.core.fitter import TimePDEFitter
from src.problem_solvers.time_pde_solver.time_schemes import create_time_scheme


class GeneralPDEFitter(TimePDEFitter):
    """General PDE Fitter with pseudo-time stepping for steady-state problems

    参考 time_pde_solver 的结构重构，使两者更加统一

    Features:
    - Pseudo-time stepping iteration (τ → ∞)
    - Automatic convergence monitoring
    - Adaptive pseudo-time step control
    - Residual-based stopping criteria
    """

    def __init__(self, config, data: Dict = None, time_scheme: str = "implicit_euler", visualizer=None):
        """Initialize General PDE Fitter

        Args:
            config: Configuration object
            data: Training data dictionary
            time_scheme: Time integration scheme (固定为 'implicit_euler' 一阶格式)
            visualizer: Visualization object
        """
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

        参考 TimePDESolver.solve_time_evolution() 的结构

        Args:
            U_init: Initial guess for solution (optional, defaults to zero)
            coeffs_init: Initial guess for coefficients (optional)

        Returns:
            Tuple: (final solution, final coefficients, convergence info)
        """
        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        # Store model reference for training
        self.model = model

        # 1. Initialize pseudo-time stepping (类似 time_pde 的 _initialize_time_evolution)
        it, tau, dt, U, U_seg, coeffs = self._initialize_pseudo_time(U_init, coeffs_init)

        # Record start time
        start_time = time.time()

        # 2. Main pseudo-time stepping loop (类似 time_pde 的 while T < T_final)
        while it < self.max_iterations:
            # 2.1 Compute adaptive time step (类似 time_pde 的 _compute_adaptive_timestep)
            dt = self._compute_adaptive_pseudo_dt(it, tau, dt, U)

            # 2.1.5 Train neural network at current step (类似 time_pde 的 _train_neural_network_step)
            if self.model is not None:
                self._train_neural_network_step(it, dt, U)

            # 2.2 Execute one pseudo-time step (直接调用父类的 time_step)
            try:
                U, U_seg, coeffs = self.time_step(
                    U, U_seg, dt,
                    coeffs_n=coeffs,
                    current_time=tau,
                    step=it
                )
            except Exception as e:
                print(f"\nError at iteration {it}: {e}")
                print("Reducing time step and retrying...")
                dt *= self.dt_decrease_factor
                dt = max(dt, self.dt_min)
                continue

            # 2.3 Update pseudo-time
            tau += dt
            it += 1

            # 2.4 Monitor solution and check convergence (类似 time_pde 的 _monitor_solution)
            converged, conv_info = self._monitor_pseudo_time_step(it, tau, dt, U, coeffs, start_time)

            # 2.5 Update animation data (类似 time_pde)
            if self.visualizer is not None:
                animation_skip = getattr(self.config, 'animation_skip', 1)
                residual = self.residual_history[-1] if self.residual_history else 0.0
                self.visualizer.update_animation_data(
                    it, residual, U, self.data, animation_skip
                )

            # 2.6 Check termination conditions
            if converged:
                return U, coeffs, conv_info

        # Maximum iterations reached
        elapsed = time.time() - start_time
        print(f"\n✗ Maximum iterations ({self.max_iterations}) reached without convergence")
        if self.residual_history:
            print(f"  Final residual: {self.residual_history[-1]:.6e}")
        print(f"  Total time: {elapsed:.2f}s")

        convergence_info = {
            "converged": False,
            "reason": "max_iterations",
            "iterations": it,
            "final_residual": self.residual_history[-1] if self.residual_history else None,
            "elapsed_time": elapsed,
            "residual_history": self.residual_history,
            "dt_history": self.dt_history,
            "solution_norm_history": self.solution_norm_history
        }

        return U, coeffs, convergence_info

    def _initialize_pseudo_time(
        self,
        U_init: Optional[np.ndarray],
        coeffs_init: Optional[np.ndarray]
    ) -> Tuple[int, float, float, np.ndarray, List[np.ndarray], np.ndarray]:
        """Initialize pseudo-time stepping parameters

        参考 TimePDESolver._initialize_time_evolution()

        Returns:
            Tuple: (iteration, tau, dt, U, U_seg, coeffs)
        """
        it = 0
        tau = 0.0
        dt = self.dt_init

        # Initialize solution
        if U_init is None:
            U_init = np.zeros((len(self.data["x"]), self.config.n_eqs))

        U = U_init.copy()
        U_seg = self.global_to_segments(U)

        # Initialize coefficients
        if coeffs_init is None:
            coeffs = self._initialize_coefficients()
        else:
            coeffs = coeffs_init.copy()

        self._coeffs_current = coeffs  # For residual computation

        # Print initialization info
        print("=" * 70)
        print("Starting Pseudo-Time Stepping Iteration")
        print("=" * 70)
        print(f"Max iterations: {self.max_iterations}")
        print(f"Convergence tolerance: {self.convergence_tol:.2e}")
        print(f"Residual tolerance: {self.residual_tol:.2e}")
        print(f"Initial dt: {dt:.6f}")
        print("=" * 70)

        return it, tau, dt, U, U_seg, coeffs

    def _compute_adaptive_pseudo_dt(
        self,
        it: int,
        tau: float,
        dt_current: float,
        U: np.ndarray
    ) -> float:
        """Compute adaptive pseudo-time step

        参考 TimePDESolver._compute_adaptive_timestep()

        Args:
            it: Current iteration
            tau: Current pseudo-time
            dt_current: Current time step
            U: Current solution

        Returns:
            Adapted time step
        """
        if not self.adaptive_dt or len(self.residual_history) < 2:
            return dt_current

        residual_current = self.residual_history[-1]
        residual_prev = self.residual_history[-2]

        # Adaptive strategy based on residual change
        if residual_current < residual_prev * 0.8:
            # Good progress, increase time step
            dt = min(dt_current * self.dt_increase_factor, self.dt_max)
        elif residual_current > residual_prev * 1.2:
            # Poor progress, decrease time step
            dt = max(dt_current * self.dt_decrease_factor, self.dt_min)
        else:
            # Maintain current time step
            dt = dt_current

        return dt

    def _monitor_pseudo_time_step(
        self,
        it: int,
        tau: float,
        dt: float,
        U: np.ndarray,
        coeffs: np.ndarray,
        start_time: float
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Monitor pseudo-time step and check convergence

        参考 TimePDESolver._monitor_solution()

        Args:
            it: Current iteration
            tau: Current pseudo-time
            dt: Current time step
            U: Current solution
            coeffs: Current coefficients
            start_time: Solver start time

        Returns:
            Tuple: (converged flag, convergence info if converged)
        """
        # Update coefficients for residual computation
        self._coeffs_current = coeffs

        # Compute residual norm
        residual = self._compute_residual_norm(U)
        solution_norm = np.linalg.norm(U)

        # Store history
        self.residual_history.append(residual)
        self.dt_history.append(dt)
        self.solution_norm_history.append(solution_norm)

        # Print progress
        if it % self.print_interval == 0:
            elapsed = time.time() - start_time
            print(f"Iter {it:6d} | τ={tau:10.6f} | dt={dt:.6e} | "
                  f"Residual={residual:.6e} | ||U||={solution_norm:.6e} | "
                  f"Time={elapsed:.2f}s")

        # Check convergence (every check_interval steps)
        if it % self.check_interval == 0:
            # Check residual convergence
            if residual < self.convergence_tol:
                elapsed = time.time() - start_time
                print("\n" + "=" * 70)
                print(f"✓ Converged at iteration {it}!")
                print(f"  Final residual: {residual:.6e}")
                print(f"  Final τ: {tau:.6f}")
                print(f"  Total time: {elapsed:.2f}s")
                print("=" * 70)

                convergence_info = {
                    "converged": True,
                    "iterations": it,
                    "final_residual": residual,
                    "final_tau": tau,
                    "elapsed_time": elapsed,
                    "residual_history": self.residual_history,
                    "dt_history": self.dt_history,
                    "solution_norm_history": self.solution_norm_history
                }
                return True, convergence_info

            # Check for divergence
            if residual > self.max_residual:
                print(f"\n✗ Divergence detected at iteration {it}!")
                print(f"  Residual: {residual:.6e} > max_residual: {self.max_residual:.6e}")
                convergence_info = {
                    "converged": False,
                    "reason": "divergence",
                    "iterations": it,
                    "final_residual": residual
                }
                return True, convergence_info  # Terminate

            # Check for stagnation
            if len(self.residual_history) > self.residual_stagnation_steps:
                recent_residuals = self.residual_history[-self.residual_stagnation_steps:]
                residual_change = max(recent_residuals) - min(recent_residuals)
                if residual_change < self.convergence_tol * 0.1:
                    print(f"\n✗ Stagnation detected at iteration {it}!")
                    print(f"  No progress for {self.residual_stagnation_steps} steps")
                    convergence_info = {
                        "converged": False,
                        "reason": "stagnation",
                        "iterations": it,
                        "final_residual": residual
                    }
                    return True, convergence_info  # Terminate

        return False, None  # Not converged, continue

    def _compute_residual_norm(self, U: np.ndarray) -> float:
        """Compute norm of PDE residual R(U)

        Args:
            U: Current solution

        Returns:
            ||R(U)||_2 / n_points
        """
        # 方法1: 使用解的变化量 (当前实现)
        # residual = ||U_new - U_old|| / max(||U||, 1)

        # 方法2: 使用PDE残差 (更物理)
        try:
            R = self.compute_residual(U)
            residual_norm = np.linalg.norm(R) / len(U)
            return residual_norm
        except:
            # Fallback: use solution change
            if len(self.solution_norm_history) > 0:
                prev_norm = self.solution_norm_history[-1]
                curr_norm = np.linalg.norm(U)
                return abs(curr_norm - prev_norm) / max(curr_norm, 1.0)
            else:
                return np.linalg.norm(U)

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
        """Neural network training at current pseudo-time step

        参考 TimePDESolver._train_neural_network_step()
        在每个伪时间步都训练神经网络，从上次的权重继承

        Args:
            it: Current iteration number
            dt: Current pseudo-time step
            U_current: Current solution for data preparation
        """
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
