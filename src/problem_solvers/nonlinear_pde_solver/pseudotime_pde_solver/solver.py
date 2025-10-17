import numpy as np
import os
import sys
import time
from typing import Optional, Dict, Any, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..solver import NonlinearPDESolver
from .core.fitter import PseudoTimePDEFitter
from .core.net import PseudoTimePDENet
from .utils.config import PseudoTimePDEConfig
from .utils.visualize import PseudoTimePDEVisualizer
from .utils.data import PseudoTimePDEDataGenerator


class PseudoTimePDESolver(NonlinearPDESolver):
    DEFAULT_TIME_SCHEME = 'implicit_euler'

    def _initialize_components(self):
        super()._initialize_components()
        self.fitter.visualizer = self.visualizer

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict[str, Any]:
        U = self._initialize_from_config() if U_init is None else np.asarray(U_init).copy()
        self.fitter.fitter_init(self.model)

        start_time = time.time()
        dt = self.fitter.dt_init
        tau = 0.0
        coeffs = self.fitter._initialize_coefficients()

        U_final, coeffs_final, conv_info = self._solve_iterations(U, dt, tau, coeffs, start_time)

        # 准备可视化数据
        U_sniper_test = self._interpolate_to_test_grid(self._to_numpy_cpu(U_final))
        U_spotter = self._get_network_prediction()
        U_spotter_test = self._interpolate_to_test_grid(self._to_numpy_cpu(U_spotter))

        results = {
            "solution": U_final,
            "coefficients": coeffs_final,
            "convergence_info": conv_info,
            "config": self.config,
            "predictions": {
                'sniper': U_sniper_test,
                'spotter': U_spotter_test
            }
        }

        self.visualizer.render(results, self.data_train, self.data_test, self.reference_solution)
        return results

    def _solve_iterations(self, U, dt, tau, coeffs, start_time) -> Tuple[np.ndarray, np.ndarray, Dict]:
        residual = float('inf')
        residual_history = []
        tau_history = []

        for iteration in range(self.fitter.max_iterations):
            self._train_neural_network_step(iteration, dt, U)
            U_from_spotter = self._get_network_prediction()
            U_seg = self.fitter.global_to_segments(U_from_spotter)

            U, _, coeffs = self.fitter.time_step(
                U_from_spotter, U_seg, dt,
                coeffs_n=coeffs,
                current_time=tau,
                step=iteration
            )

            residual, _ = self.fitter._compute_residual_norm(U, U_from_spotter)
            residual_history.append(residual)
            tau_history.append(tau)
            tau += dt
            dt, _ = self.fitter._update_adaptive_timestep(residual, residual, 0)

        return U, coeffs, {
            'converged': residual < self.fitter.residual_tol,
            'iterations': self.fitter.max_iterations,
            'final_residual': residual,
            'final_tau': tau,
            'elapsed_time': time.time() - start_time,
            'residual_history': residual_history,
            'tau_history': tau_history
        }

    def _create_config(self, case_dir: str):
        return PseudoTimePDEConfig(case_dir=case_dir)

    def _create_data_generator(self):
        return PseudoTimePDEDataGenerator(self.config)

    def _create_model(self):
        return PseudoTimePDENet.model_init(self.config)

    def _create_fitter(self, time_scheme: Optional[str]):
        return PseudoTimePDEFitter(self.config, self.data_train, self.visualizer)

    def _create_visualizer(self):
        return PseudoTimePDEVisualizer(self.config)

    def _get_default_time_scheme(self) -> Optional[str]:
        return self.DEFAULT_TIME_SCHEME

    def _process_reference_solution(self, data: Dict, x_ref: np.ndarray, mat_file: str) -> Dict:
        from scipy.io import loadmat
        mat_data = loadmat(mat_file)
        u_ref = mat_data['usol'].T.flatten()
        x_domain = self.config.x_domain
        x_ref = np.linspace(x_domain[0][0], x_domain[0][1], mat_data['x'].size)
        y_ref = np.linspace(x_domain[1][0], x_domain[1][1], mat_data['t'].size)
        return {'x_ref': x_ref, 'y_ref': y_ref, 'u_ref': u_ref}
