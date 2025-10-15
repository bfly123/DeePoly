"""
Pseudo-Time PDE Fitter with Pseudo-Time Stepping

Solves steady-state PDEs using pseudo-time stepping:
    ∂U/∂τ + R(U) = 0 → U as τ → ∞

Now directly inherits from BaseDeepPolyFitter (no dependency on TimePDEFitter).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver


class PseudoTimePDEFitter(BaseDeepPolyFitter):
    """Pseudo-Time PDE Fitter with pseudo-time stepping for steady-state problems

    Features:
    - Pseudo-time stepping iteration
    - Automatic convergence monitoring
    - Adaptive pseudo-time step control
    - Residual-based stopping criteria
    """

    # Adaptive time-stepping thresholds
    ADAPTIVE_DT_THRESHOLD_HIGH = 0.1      # Above this: use dt_max
    ADAPTIVE_DT_THRESHOLD_MID = 0.01      # Above this: interpolate between 0.2*dt_max and dt_max
    ADAPTIVE_DT_THRESHOLD_LOW = 1e-4      # Above this: interpolate between dt_min and 0.2*dt_max
    ADAPTIVE_DT_SCALE_FACTOR = 0.2        # Scale factor for mid-range dt
    ADAPTIVE_DT_INTERP_HIGH = 0.8         # Interpolation weight for high range

    # Convergence checking
    CONVERGENCE_CHECK_FACTOR = 10         # Check convergence more frequently when residual < tol * this factor

    # Initialization
    COEFF_PERTURBATION = 1e-8             # Random perturbation to break symmetry

    # Neural network training defaults
    NN_DEFAULT_MAX_RETRIES = 3
    NN_DEFAULT_LEARNING_RATE = 0.001
    NN_DEFAULT_EPOCHS_ADAM = 100
    NN_DEFAULT_EPOCHS_LBFGS = 0
    NN_DEFAULT_TOL = 1e-6

    # Error handling
    MAX_CONSECUTIVE_FAILURES = 5          # Max consecutive failures before giving up

    def __init__(self, config, data: Dict = None, time_scheme: str = "implicit_euler", visualizer=None):
        # Initialize base class with time_scheme support (like TimePDEFitter)
        super().__init__(config, dt=None, data=data, time_scheme=time_scheme)

        # Initialize solver (needed since we no longer inherit from TimePDEFitter)
        self.solver = LinearSolver(
            verbose=True, use_gpu=True, performance_tracking=True
        )

        # Pseudo-time stepping parameters
        self.max_iterations = config.max_iterations
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

        # Cache segment indices for performance (computed once instead of repeatedly)
        self._segment_start_indices = None
        self._segment_end_indices = None

    def solve_steady_state(
        self,
        U_init: Optional[np.ndarray] = None,
        coeffs_init: Optional[np.ndarray] = None,
        model: Optional[object] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Solve steady-state PDE using pseudo-time stepping

        DEPRECATED: This method is no longer used. The solver layer now manages
        the iteration loop and DNN training. This method is kept for backward
        compatibility but should not be called directly.

        Args:
            U_init: Initial guess for solution (optional, defaults to zero)
            coeffs_init: Initial guess for coefficients (optional)
            model: Neural network model (unused, kept for compatibility)

        Returns:
            Tuple: (final solution, final coefficients, convergence info)
        """
        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        # Pre-compute segment indices for performance (done once)
        if self._segment_start_indices is None:
            self._cache_segment_indices()

        # Note: model parameter is ignored - training now handled by solver layer

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
        U_current = U_init  # No copy needed - U_init not reused
        U_seg_current = U_seg  # No copy needed - U_seg not reused
        coeffs_current = coeffs_init  # No copy needed - coeffs_init not reused

        residual_prev = None
        stagnation_count = 0
        consecutive_failures = 0

        print("\n=== Pseudo-Time Stepping ===")
        print(f"Target tolerance: {self.residual_tol:.2e} | Max iter: {self.max_iterations}")

        start_time = time.time()

        # Pseudo-time stepping loop
        # NOTE: Neural network training is now managed by the solver layer
        for iteration in range(self.max_iterations):
            # Execute one time step with improved error handling
            try:
                U_new, U_seg_new, coeffs_new = self.time_step(
                    U_current, U_seg_current, dt,
                    coeffs_n=coeffs_current,
                    current_time=tau,
                    step=iteration
                )
                consecutive_failures = 0  # Reset on success
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                consecutive_failures += 1
                print(f"  Error at iter {iteration} (attempt {consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES}): "
                      f"{type(e).__name__}: {str(e)[:50]}...")

                if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    print(f"\n✗ Solver failed after {consecutive_failures} consecutive errors")
                    elapsed = time.time() - start_time
                    info = self._build_convergence_info(False, "consecutive_failures", iteration + 1,
                                                        float('inf'), tau, elapsed)
                    return U_current, coeffs_current, info

                # Reduce time step and retry
                dt = max(dt * self.dt_decrease_factor, self.dt_min)
                continue
            except Exception as e:
                # Unexpected error - log and abort
                print(f"\n✗ Unexpected error at iter {iteration}: {type(e).__name__}: {str(e)}")
                elapsed = time.time() - start_time
                info = self._build_convergence_info(False, "unexpected_error", iteration + 1,
                                                    float('inf'), tau, elapsed)
                return U_current, coeffs_current, info

            # Compute residual and solution norm
            residual, solution_norm = self._compute_residual_norm(U_new, U_current)

            # Update history and pseudo-time
            self._update_convergence_history(residual, dt, solution_norm)
            tau += dt

            # Update animation data
            if self.visualizer is not None:
                animation_skip = getattr(self.config, 'animation_skip', 1)
                self.visualizer.update_animation_data(
                    iteration + 1, residual, U_new, self.data, animation_skip
                )

            # Store dt before adaptive adjustment for printing
            dt_used = dt
            decrease_rate = residual / residual_prev if residual_prev is not None and residual_prev > 0 else 1.0

            # Check convergence
            conv_msg = self._check_convergence(iteration, residual, tau, start_time)
            if conv_msg is not None:
                print(conv_msg)
                elapsed = time.time() - start_time
                info = self._build_convergence_info(True, "", iteration + 1, residual, tau, elapsed)
                return U_new, coeffs_new, info

            # Adaptive time step control
            dt, stagnation_count = self._update_adaptive_timestep(residual, residual_prev, stagnation_count)

            # Check for divergence or stagnation
            failure = self._check_failure(residual, stagnation_count, iteration)
            if failure is not None:
                msg, reason = failure
                print(msg)
                elapsed = time.time() - start_time
                info = self._build_convergence_info(False, reason, iteration + 1, residual, tau, elapsed)
                return U_new, coeffs_new, info

            # Print progress
            elapsed = time.time() - start_time
            self._print_iteration_progress(iteration, tau, dt_used, dt, residual, decrease_rate, elapsed)

            # Update for next iteration (no copy needed - time_step returns new arrays)
            U_current = U_new
            U_seg_current = U_seg_new
            coeffs_current = coeffs_new
            self._coeffs_current = coeffs_current
            residual_prev = residual

        # Maximum iterations reached
        elapsed = time.time() - start_time
        print(f"\n✗ Max iterations reached: Res={residual:.2e} | Time={elapsed:.1f}s")
        info = self._build_convergence_info(False, "max_iterations", self.max_iterations, residual, tau, elapsed)
        return U_current, coeffs_current, info

    def _initialize_coefficients(self) -> np.ndarray:
        """Initialize coefficient array with zeros or small random values"""
        coeffs = np.zeros((self.ns, self.config.n_eqs, self.dgN))
        # Add small random perturbation to avoid exact symmetry
        coeffs += np.random.randn(*coeffs.shape) * self.COEFF_PERTURBATION
        return coeffs

    def compute_residual(self, U: np.ndarray) -> np.ndarray:
        """Compute residual R(U) = L1(U) + L2(U)·F(U) + N(U) - S"""
        if not self._precompiled or not hasattr(self, '_coeffs_current') or self._coeffs_current is None:
            raise RuntimeError("Operators not precompiled or coefficients not available")

        U_seg = self.global_to_segments(U)
        residual = np.zeros_like(U)

        for seg_idx in range(self.ns):
            features = self._features[seg_idx][0]
            coeffs = self._coeffs_current[seg_idx]
            L1_ops = self._linear_operators[seg_idx].get("L1", None)
            L2_ops = self._linear_operators[seg_idx].get("L2", None)

            F_val = self.F_func(features, U_seg[seg_idx])
            N_val = self.N_func(features, U_seg[seg_idx])
            S_val = self._get_source_term(seg_idx)

            residual_seg = np.zeros_like(U_seg[seg_idx])
            for eq_idx in range(self.config.n_eqs):
                if L1_ops is not None and eq_idx < len(L1_ops):
                    residual_seg[:, eq_idx] += L1_ops[eq_idx] @ coeffs[eq_idx]
                if L2_ops is not None and eq_idx < len(L2_ops):
                    residual_seg[:, eq_idx] += (L2_ops[eq_idx] @ coeffs[eq_idx]) * F_val[:, eq_idx]
                residual_seg[:, eq_idx] += N_val[eq_idx] - S_val[:, eq_idx]

            start_idx, end_idx = self._get_segment_global_indices(seg_idx)
            residual[start_idx:end_idx] = residual_seg

        return residual

    def _cache_segment_indices(self) -> None:
        """Pre-compute and cache segment global indices (O(n) → O(1) lookups)"""
        self._segment_start_indices = []
        self._segment_end_indices = []
        cumsum = 0
        for seg in self.data["x_segments_norm"]:
            self._segment_start_indices.append(cumsum)
            cumsum += len(seg)
            self._segment_end_indices.append(cumsum)

    def _get_segment_global_indices(self, segment_idx: int) -> Tuple[int, int]:
        """Get global array indices for a segment (O(1) cached lookup)"""
        return self._segment_start_indices[segment_idx], self._segment_end_indices[segment_idx]

    def _compute_residual_norm(self, U_new: np.ndarray, U_current: np.ndarray) -> Tuple[float, float]:
        """Compute normalized residual and solution norm"""
        delta_U = U_new - U_current
        u_norm = np.linalg.norm(U_new)
        u_current_norm = np.linalg.norm(U_current)
        residual = np.linalg.norm(delta_U) / max(u_norm, u_current_norm, 1.0)
        return residual, u_norm

    def _update_convergence_history(self, residual: float, dt: float, solution_norm: float) -> None:
        """Update convergence history lists"""
        self.residual_history.append(residual)
        self.dt_history.append(dt)
        self.solution_norm_history.append(solution_norm)

    def _build_convergence_info(self, converged: bool, reason: str, iterations: int,
                                residual: float, tau: float, elapsed: float) -> Dict[str, Any]:
        """Build convergence information dictionary"""
        info = {
            "converged": converged,
            "iterations": iterations,
            "final_residual": residual,
            "elapsed_time": elapsed
        }
        if converged:
            info["final_tau"] = tau
            info["residual_history"] = self.residual_history
            info["dt_history"] = self.dt_history
            info["solution_norm_history"] = self.solution_norm_history
        else:
            info["reason"] = reason
            if reason == "max_iterations":
                info["residual_history"] = self.residual_history
                info["dt_history"] = self.dt_history
                info["solution_norm_history"] = self.solution_norm_history
        return info

    def _check_convergence(self, iteration: int, residual: float, tau: float,
                          start_time: float) -> Optional[str]:
        """Check convergence and return success message if converged, None otherwise"""
        should_check = (iteration + 1) % self.check_interval == 0 or residual < self.residual_tol * self.CONVERGENCE_CHECK_FACTOR
        if should_check and residual < self.residual_tol:
            elapsed = time.time() - start_time
            return f"\n✓ Converged at iter {iteration+1}: Res={residual:.2e} | τ={tau:.4f} | Time={elapsed:.1f}s"
        return None

    def _update_adaptive_timestep(self, residual: float, residual_prev: Optional[float],
                                 stagnation_count: int) -> Tuple[float, int]:
        """Update adaptive time step and stagnation count"""
        if not self.adaptive_dt:
            return self.dt_init, stagnation_count

        # Adaptive dt control based on residual magnitude
        if residual > self.ADAPTIVE_DT_THRESHOLD_HIGH:
            dt = self.dt_max
        elif residual > self.ADAPTIVE_DT_THRESHOLD_MID:
            # Interpolate between 0.2*dt_max and dt_max
            dt_range = self.ADAPTIVE_DT_THRESHOLD_HIGH - self.ADAPTIVE_DT_THRESHOLD_MID
            scale = (residual - self.ADAPTIVE_DT_THRESHOLD_MID) / dt_range
            dt = self.dt_max * (self.ADAPTIVE_DT_SCALE_FACTOR + self.ADAPTIVE_DT_INTERP_HIGH * scale)
        elif residual > self.ADAPTIVE_DT_THRESHOLD_LOW:
            # Interpolate between dt_min and 0.2*dt_max
            dt_range = self.ADAPTIVE_DT_THRESHOLD_MID - self.ADAPTIVE_DT_THRESHOLD_LOW
            scale = (residual - self.ADAPTIVE_DT_THRESHOLD_LOW) / dt_range
            dt_target = self.dt_max * self.ADAPTIVE_DT_SCALE_FACTOR
            dt = self.dt_min + (dt_target - self.dt_min) * scale
        else:
            dt = self.dt_min

        dt = np.clip(dt, self.dt_min, self.dt_max)

        # Update stagnation count
        new_stagnation = stagnation_count + 1 if (residual_prev and residual > residual_prev) else 0
        return dt, new_stagnation

    def _check_failure(self, residual: float, stagnation_count: int, iteration: int) -> Optional[Tuple[str, str]]:
        """Check for divergence or stagnation. Returns (message, reason) if failed, None otherwise"""
        if residual > self.max_residual:
            msg = f"\n✗ Diverged at iter {iteration+1}: Res={residual:.2e} > {self.max_residual:.2e}"
            return msg, "divergence"

        if stagnation_count > self.residual_stagnation_steps:
            msg = f"\n✗ Stagnated at iter {iteration+1}: No progress for {stagnation_count} steps"
            return msg, "stagnation"

        return None

    def _print_iteration_progress(self, iteration: int, tau: float, dt_used: float,
                                  dt: float, residual: float, decrease_rate: float,
                                  elapsed: float) -> None:
        """Print iteration progress line"""
        if (iteration + 1) % self.print_interval == 0 or iteration == 0:
            dt_arrow = "↑" if dt > dt_used else ("↓" if dt < dt_used else "")
            print(f"Iter {iteration+1:5d} | τ={tau:8.4f} | dτ={dt_used:.2e}{dt_arrow} | "
                  f"Res={residual:.2e}(×{decrease_rate:.2f}) | {elapsed:.1f}s")


    def _build_segment_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build segment Jacobian matrix using the time scheme

        Delegates to the configured time integration scheme for time-specific handling.
        This implements the abstract method from BaseDeepPolyFitter.
        """
        if self.time_scheme is None:
            raise RuntimeError("Time scheme not initialized")
        return self.time_scheme.build_stage_jacobian(segment_idx, **kwargs)