"""
Pseudo-Time PDE Solver with Pseudo-Time Stepping

Solves arbitrary steady-state PDEs using pseudo-time stepping iteration.
"""

import numpy as np
import os
import sys
import time
from typing import Optional, Dict, Any, Tuple

# Ensure project modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..solver import NonlinearPDESolver
from .core.fitter import PseudoTimePDEFitter
from .core.net import PseudoTimePDENet
# SpotterNet is imported dynamically via _get_spotter_net_class when needed
from .utils.config import PseudoTimePDEConfig
from .utils.visualize import PseudoTimePDEVisualizer
from .utils.data import PseudoTimePDEDataGenerator


class PseudoTimePDESolver(NonlinearPDESolver):
    """Pseudo-Time PDE solver using pseudo-time stepping for steady-state problems

    Now inherits from NonlinearPDESolver for unified DNN training management.

    Features:
    - Unified treatment of all derivatives
    - Pseudo-time stepping to steady state
    - Automatic convergence monitoring
    - Adaptive time step control
    - DNN training managed at solver layer
    """

    def _initialize_config(self, config, case_dir):
        """Initialize configuration using base workflow"""
        super()._initialize_config(config, case_dir)

    def _initialize_components(self):
        """Initialize core components via base implementation"""
        super()._initialize_components()

        # Ensure fitter can access visualizer for animation hooks
        if getattr(self.fitter, "visualizer", None) is None:
            self.fitter.visualizer = self.visualizer

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Solve steady-state PDE using pseudo-time stepping

        Returns:
            Dict with 'solution', 'coefficients', 'convergence_info', 'config'
        """
        self._print_header()

        # Step 1: Initialize solution
        U_init = self._setup_initial_solution(U_init)

        # Step 2: Initialize fitter and prepare iteration state
        state = self._initialize_iteration_state(U_init)

        # Step 3: Run pseudo-time iteration
        print("\nStep 3: Pseudo-Time Iteration")
        print("-" * 70)
        print(f"Target tolerance: {self.fitter.residual_tol:.2e} | "
              f"Max iter: {self.fitter.max_iterations} | "
              f"Initial dτ: {state['dt']:.2e}")

        U_final, coeffs_final, conv_info = self.solve_steady_state(state)

        # Step 4: Finalize and post-process
        self._finalize_solve(U_final, coeffs_final, conv_info)

        return {
            "solution": U_final,
            "coefficients": coeffs_final,
            "convergence_info": conv_info,
            "config": self.config
        }

    def _print_header(self):
        """Print solver information header"""
        print("\n" + "=" * 70)
        print("Pseudo-Time PDE Solver - Pseudo-Time Stepping")
        print("=" * 70)
        print(f"Problem: {getattr(self.config, 'problem_name', 'Steady-State PDE')}")
        print(f"Domain: {self.config.x_domain} | "
              f"Segments: {self.config.n_segments} | "
              f"Poly degree: {self.config.poly_degree}")
        print(f"Number of equations: {self.config.n_eqs}")
        print("=" * 70 + "\n")

    def _setup_initial_solution(self, U_init: Optional[np.ndarray]) -> np.ndarray:
        """Initialize or validate initial solution"""
        initial_condition = getattr(self.config, 'initial_condition', {})
        init_method = initial_condition.get('method', 'zero') if isinstance(initial_condition, dict) else 'zero'

        print(f"Step 1: Initialize solution (method='{init_method}')")
        print("-" * 70)

        if U_init is None:
            U_init = self._initialize_from_config()
        else:
            U_init = np.asarray(U_init).copy()
            print(f"  Using provided initial guess: shape {U_init.shape}")

        return U_init

    def _initialize_iteration_state(self, U_init: np.ndarray) -> Dict[str, Any]:
        """Prepare fitter and create initial iteration state"""
        print("\nStep 2: Initialize Fitter")
        print("-" * 70)

        # Initialize fitter operators
        self.fitter.fitter_init(self.model)
        if getattr(self.fitter, "_segment_start_indices", None) is None:
            self.fitter._cache_segment_indices()
        print("  ✓ Operators precompiled and cached")

        # Reset convergence tracking
        self.fitter.residual_history = []
        self.fitter.dt_history = []
        self.fitter.solution_norm_history = []

        # Create initial state dictionary
        return {
            'U': U_init,
            'U_seg': self.fitter.global_to_segments(U_init),
            'coeffs': self.fitter._initialize_coefficients(),
            'dt': self.fitter.dt_init,
            'tau': 0.0,
            'residual_prev': None,
            'stagnation_count': 0,
            'consecutive_failures': 0,
            'start_time': time.time()
        }

    def solve_steady_state(self, state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Solve steady-state PDE using pseudo-time stepping

        Returns:
            (U_final, coeffs_final, convergence_info)
        """
        for iteration in range(self.fitter.max_iterations):
            # Train DNN and execute one pseudo-time step
            success, state = self._perform_iteration_step(iteration, state)

            if not success:
                # Iteration failed, check if we should abort or retry
                if state['consecutive_failures'] >= self.fitter.MAX_CONSECUTIVE_FAILURES:
                    conv_info = self._build_failure_info(state, "consecutive_failures", iteration)
                    print("\n✗ Solver aborted due to repeated failures")
                    return state['U'], state['coeffs'], conv_info

                # Reduce dt and retry
                state['dt'] = max(state['dt'] * self.fitter.dt_decrease_factor, self.fitter.dt_min)
                continue

            # Check convergence and update state
            conv_status = self._check_and_update_state(iteration, state)

            if conv_status == 'converged':
                conv_info = self._build_success_info(state, iteration)
                print(f"\n✓ Converged at iter {iteration+1}: Res={state['residual']:.2e}")
                return state['U'], state['coeffs'], conv_info
            elif conv_status == 'failed':
                conv_info = self._build_failure_info(state, state['failure_reason'], iteration)
                return state['U'], state['coeffs'], conv_info

        # Max iterations reached
        conv_info = self._build_failure_info(state, "max_iterations", self.fitter.max_iterations - 1)
        residual = state.get('residual_prev', float('inf'))
        print(f"\n✗ Max iterations ({self.fitter.max_iterations}) reached: Res={residual:.2e}")
        return state['U'], state['coeffs'], conv_info

    def _perform_iteration_step(self, iteration: int, state: Dict) -> Tuple[bool, Dict]:
        """Perform one iteration: DNN training + time step

        Returns:
            (success, updated_state)
        """
        # Train neural network
        self._train_neural_network_step(iteration, state['dt'], state['U'])

        # Refresh segment indices after fitter re-initialization
        if getattr(self.fitter, "_segment_start_indices", None) is None:
            self.fitter._cache_segment_indices()
        state['U_seg'] = self.fitter.global_to_segments(state['U'])

        # Execute time step with error handling
        try:
            U_new, U_seg_new, coeffs_new = self.fitter.time_step(
                state['U'], state['U_seg'], state['dt'],
                coeffs_n=state['coeffs'],
                current_time=state['tau'],
                step=iteration
            )
            state.update({'U': U_new, 'U_seg': U_seg_new, 'coeffs': coeffs_new})
            state['consecutive_failures'] = 0
            return True, state

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            state['consecutive_failures'] += 1
            print(f"  Error at iter {iteration} (attempt {state['consecutive_failures']}/"
                  f"{self.fitter.MAX_CONSECUTIVE_FAILURES}): {type(exc).__name__}")
            return False, state

        except Exception as exc:
            print(f"\n✗ Unexpected error at iter {iteration}: {type(exc).__name__}: {exc}")
            state['consecutive_failures'] = self.fitter.MAX_CONSECUTIVE_FAILURES
            state['failure_reason'] = "unexpected_error"
            return False, state

    def _check_and_update_state(self, iteration: int, state: Dict) -> str:
        """Update iteration state and check convergence/failure

        Returns:
            'converged', 'failed', or 'continue'
        """
        # Compute residual and update histories
        residual, solution_norm = self.fitter._compute_residual_norm(state['U'],
                                    state.get('U_prev', state['U']))
        self.fitter._update_convergence_history(residual, state['dt'], solution_norm)
        self.fitter._coeffs_current = state['coeffs']

        state['residual'] = residual
        state['tau'] += state['dt']
        dt_used = state['dt']

        # Update visualization
        if self.visualizer is not None:
            animation_skip = getattr(self.config, 'animation_skip', 1)
            self.visualizer.update_animation_data(
                iteration + 1, residual, state['U'], self.data_train, animation_skip
            )

        # Check convergence
        if residual < self.fitter.residual_tol:
            return 'converged'

        # Check failure conditions
        failure = self.fitter._check_failure(residual, state['stagnation_count'], iteration)
        if failure is not None:
            msg, reason = failure
            print(msg)
            state['failure_reason'] = reason
            return 'failed'

        # Update adaptive time step
        state['dt'], state['stagnation_count'] = self.fitter._update_adaptive_timestep(
            residual, state['residual_prev'], state['stagnation_count']
        )

        # Print progress
        decrease_rate = (residual / state['residual_prev']
                        if state['residual_prev'] is not None and state['residual_prev'] > 0
                        else 1.0)
        elapsed = time.time() - state['start_time']
        self.fitter._print_iteration_progress(
            iteration, state['tau'], dt_used, state['dt'], residual, decrease_rate, elapsed
        )

        # Update state for next iteration
        state['U_prev'] = state['U'].copy()
        state['residual_prev'] = residual

        return 'continue'

    def _build_success_info(self, state: Dict, iteration: int) -> Dict:
        """Build convergence info for successful solve"""
        elapsed = time.time() - state['start_time']
        return self.fitter._build_convergence_info(
            True, "", iteration + 1, state['residual'], state['tau'], elapsed
        )

    def _build_failure_info(self, state: Dict, reason: str, iteration: int) -> Dict:
        """Build convergence info for failed solve"""
        elapsed = time.time() - state['start_time']
        residual = state.get('residual', state.get('residual_prev', float('inf')))
        return self.fitter._build_convergence_info(
            False, reason, iteration + 1, residual, state['tau'], elapsed
        )

    def _finalize_solve(self, U_final: np.ndarray, coeffs_final: np.ndarray, conv_info: Dict):
        """Finalize visualization and post-processing"""
        if getattr(self.config, 'realtime_visualization', False):
            self.visualizer.finalize_realtime_animation()

        if conv_info.get("converged"):
            print("\nStep 4: Post-Processing")
            print("-" * 70)
            self._postprocess_results(U_final, coeffs_final, conv_info)

    def _process_reference_solution(self, data: Dict, x_ref: np.ndarray, mat_file: str) -> Optional[Dict]:
        """Process reference solution data specific to pseudo-time solver"""
        y_ref = None
        y_label = None
        if 'y' in data:
            y_ref = data['y'].flatten()
            y_label = 'y'
        elif 't' in data:
            y_ref = data['t'].flatten()
            y_label = 't'
            print("  Using time 't' as second spatial dimension 'y'")

        if 'usol' in data:
            u_ref = data['usol']
        elif 'u' in data:
            u_ref = data['u']
        else:
            print("Warning: Solution data not found in reference file")
            return None

        u_ref = np.asarray(u_ref)
        if u_ref.ndim == 2:
            if y_ref is not None:
                if u_ref.shape == (len(y_ref), len(x_ref)):
                    u_ref = u_ref.flatten()
                    print(f"  2D solution shape: ({len(y_ref)}, {len(x_ref)}) -> flattened")
                elif u_ref.shape == (len(x_ref), len(y_ref)):
                    u_ref = u_ref.T.flatten()
                    print(f"  2D solution transposed to match axes and flattened")
                else:
                    u_ref = u_ref.flatten()
                    print("  Warning: Shape mismatch, flattened reference solution")
            else:
                u_ref = u_ref[-1, :]
                print("  Extracted final time slice as steady-state reference")
        elif u_ref.ndim > 2:
            u_ref = u_ref.reshape(-1)
            print("  Warning: Higher-dimensional reference data flattened")

        print(f"  Spatial points: {len(x_ref)}")
        if y_ref is not None:
            print(
                f"  Secondary axis points: {len(y_ref)} | "
                f"Ranges: x=[{x_ref.min():.6f}, {x_ref.max():.6f}], "
                f"{y_label or 'y'}=[{y_ref.min():.6f}, {y_ref.max():.6f}]"
            )
        else:
            print(f"  Spatial range: [{x_ref.min():.6f}, {x_ref.max():.6f}]")

        result = {
            'x_ref': x_ref,
            'u_ref': u_ref,
            'file_path': mat_file
        }
        if y_ref is not None:
            result['y_ref'] = y_ref
            result['y_label'] = y_label or 'y'
        return result

    def _postprocess_results(self, U_steady, coeffs_steady, conv_info):
        """Post-process solution and generate visualizations"""
        results_dir = self.config.results_dir

        U_steady_cpu = self._to_numpy_cpu(U_steady)
        coeffs_cpu = self._to_numpy_cpu(coeffs_steady)

        print("Interpolating solution to test grid...")
        U_test = self._interpolate_to_test_grid(U_steady_cpu)

        self.visualizer.save_convergence_history(conv_info, results_dir)
        self.visualizer.plot_final_solution(
            U_test,
            self.data_test,
            os.path.join(results_dir, "final_solution.png")
        )
        self.visualizer.plot_residual_history(
            os.path.join(results_dir, "residual_history.png")
        )
        self.visualizer.visualize_solution(U_test, self.data_test, results_dir)

        if self.reference_solution is not None:
            print("Generating reference solution comparison...")
            self.visualizer.plot_reference_comparison(
                U_test,
                self.data_test,
                self.reference_solution,
                os.path.join(results_dir, "reference_comparison.png")
            )

        if getattr(self.config, 'save_animation', True):
            self.visualizer.create_iteration_gif(
                os.path.join(results_dir, "iteration_evolution.gif")
            )

        self.visualizer.save_solution_data(
            U_steady_cpu,
            U_test,
            coeffs_cpu,
            conv_info,
            self.data_train,
            self.data_test,
            results_dir
        )

        print(f"✓ Results saved to: {results_dir}")

    # ==================== 时间步控制方法（PseudoTime特定） ====================

    def get_solver_time_info(self) -> Dict[str, Any]:
        """获取伪时间信息 - PseudoTime特定覆盖

        Returns:
            包含伪时间迭代信息的字典
        """
        info = super().get_solver_time_info()

        # Dynamically get the actual time scheme from fitter
        scheme_name = 'N/A'
        if hasattr(self.fitter, 'time_scheme') and self.fitter.time_scheme is not None:
            scheme_info = self.fitter.time_scheme.get_scheme_info()
            scheme_name = scheme_info.get('method', 'implicit_euler')
        elif hasattr(self.fitter, '_time_scheme_name'):
            scheme_name = self.fitter._time_scheme_name
        else:
            scheme_name = 'implicit_euler'  # Default fallback

        info.update({
            "time_type": "pseudo_time",
            "scheme": scheme_name,
            "adaptive": True,
            "residual_based": True,
            "convergence_criterion": "residual_tolerance",
            "max_iterations": getattr(self.fitter, 'max_iterations', 'N/A'),
            "residual_tolerance": getattr(self.fitter, 'residual_tol', 'N/A')
        })
        return info

    # Factory methods required by NonlinearPDESolver
    def _create_config(self, case_dir: str):
        return PseudoTimePDEConfig(case_dir=case_dir)

    def _create_data_generator(self):
        return PseudoTimePDEDataGenerator(self.config)

    def _create_model(self):
        return PseudoTimePDENet.model_init(self.config)

    def _create_fitter(self, time_scheme: Optional[str]):
        scheme = time_scheme or 'implicit_euler'
        return PseudoTimePDEFitter(
            config=self.config,
            data=self.data_train,
            time_scheme=scheme,
            visualizer=self.visualizer
        )

    def _create_visualizer(self):
        return PseudoTimePDEVisualizer(self.config)

    def _get_default_time_scheme(self) -> Optional[str]:
        return 'implicit_euler'

    def _get_spotter_net_class(self):
        from .core.spotter_init import SpotterNet
        return SpotterNet


def main():
    """Main entry point for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Pseudo-Time PDE Solver with Pseudo-Time Stepping")
    parser.add_argument("--case_path", type=str, required=True, help="Path to case directory")
    args = parser.parse_args()

    # Create solver
    solver = PseudoTimePDESolver(case_dir=args.case_path)

    # Solve
    results = solver.solve()

    if results["convergence_info"]["converged"]:
        print("\n✓ Steady-state solution obtained successfully!")
    else:
        print(f"\n✗ Failed to converge: {results['convergence_info']['reason']}")


if __name__ == "__main__":
    main()
