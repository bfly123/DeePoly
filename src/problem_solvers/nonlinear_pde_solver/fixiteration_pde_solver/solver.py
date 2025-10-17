import numpy as np
import os
import time
from typing import Optional, Dict, Any, Tuple

from ..solver import NonlinearPDESolver
from .core.fitter import FixIterationPDEFitter
from .core.net import FixIterationPDENet
from .utils.config import FixIterationPDEConfig
from .utils.data import FixIterationPDEDataGenerator
from .utils.visualize import FixIterationPDEVisualizer


class FixIterationPDESolver(NonlinearPDESolver):

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict[str, Any]:
        U = self._initialize_from_config() if U_init is None else U_init.copy()
        self.fitter.fitter_init(self.model)

        U_final, coeffs_final, conv_info = self._solve_with_fixed_point_iteration(U)
        self._postprocess_results(U_final, coeffs_final, conv_info)

        return {
            "solution": U_final,
            "coefficients": coeffs_final,
            "convergence_info": conv_info
        }

    def _solve_with_fixed_point_iteration(self, U_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        start_time = time.time()
        max_iterations = self.fitter.max_iterations
        convergence_tol = self.fitter.convergence_tol
        print_interval = getattr(self.fitter, 'print_interval', 10)

        self._train_neural_network_step(0, 0.01, U_init)
        U_current = self._get_network_prediction()
        self._save_network_comparison()

        residual_history = []
        coeffs_current = None

        for iteration in range(max_iterations):
            coeffs_new = self.fitter.fit(U_current=U_current)
            U_new, _ = self.fitter.construct(self.data_train, self.model, coeffs_new)
            residual = np.linalg.norm(U_new - U_current) / (np.linalg.norm(U_new) + 1e-10)
            residual_history.append(residual)

            if iteration % print_interval == 0 or residual < convergence_tol:
                print(f"Iter {iteration+1:4d} | Residual: {residual:.6e} | ||U||: {np.linalg.norm(U_new):.6e}")

            if residual < convergence_tol:
                self.fitter.residual_history = residual_history
                return U_new, coeffs_new, {
                    "converged": True,
                    "iterations": iteration + 1,
                    "final_residual": residual,
                    "elapsed_time": time.time() - start_time,
                    "residual_history": residual_history
                }

            U_current = U_new.copy()
            coeffs_current = coeffs_new.copy()

        self.fitter.residual_history = residual_history
        return U_current, coeffs_current, {
            "converged": False,
            "reason": "max_iterations",
            "iterations": max_iterations,
            "final_residual": residual,
            "elapsed_time": time.time() - start_time,
            "residual_history": residual_history
        }

    def _get_network_prediction(self) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(self.data_train['x'], dtype=torch.float64, device=self.config.device)
            _, U_pred = self.model(x_tensor)
        return U_pred.cpu().numpy()

    def _postprocess_results(self, U_final, coeffs_final, conv_info):
        U_final_cpu = self._to_numpy_cpu(U_final)
        U_test = self._interpolate_to_test_grid(U_final_cpu)

        results_dir = self.config.results_dir
        self.visualizer.save_convergence_history(conv_info, results_dir)
        self.visualizer.plot_final_solution(U_test, self.data_test,
                                           os.path.join(results_dir, "final_solution.png"))
        self.visualizer.plot_residual_history(os.path.join(results_dir, "residual_history.png"))
        self._save_sniper_comparison(U_test)

    def _save_network_comparison(self):
        if not hasattr(self, 'reference_solution') or self.reference_solution is None:
            return
        U_network_train = self._get_network_prediction()
        U_network_test = self._interpolate_to_test_grid(U_network_train)
        save_path = os.path.join(self.config.results_dir, "network_vs_reference.png")
        self.visualizer.plot_reference_comparison(U_network_test, self.data_test,
                                                  self.reference_solution, save_path)

    def _save_sniper_comparison(self, U_sniper_test):
        if not hasattr(self, 'reference_solution') or self.reference_solution is None:
            return
        save_path = os.path.join(self.config.results_dir, "sniper_vs_reference.png")
        self.visualizer.plot_reference_comparison(U_sniper_test, self.data_test,
                                                  self.reference_solution, save_path)

    def get_solver_time_info(self) -> Dict[str, Any]:
        info = super().get_solver_time_info()
        info.update({
            "solver_type": "fixed_point_iteration",
            "has_time_stepping": False,
            "convergence_based": True,
            "max_iterations": getattr(self.fitter, 'max_iterations', 'N/A'),
            "convergence_tolerance": getattr(self.fitter, 'convergence_tol', 'N/A')
        })
        return info

    def _create_config(self, case_dir: str):
        return FixIterationPDEConfig(case_dir=case_dir)

    def _create_data_generator(self):
        return FixIterationPDEDataGenerator(self.config)

    def _create_model(self):
        return FixIterationPDENet.model_init(self.config)

    def _create_fitter(self, time_scheme: Optional[str]):
        return FixIterationPDEFitter(config=self.config, data=self.data_train)

    def _create_visualizer(self):
        return FixIterationPDEVisualizer(self.config)

    def _get_default_time_scheme(self) -> Optional[str]:
        return None

    def _process_reference_solution(self, data: Dict, x_ref: np.ndarray, mat_file: str) -> Dict:
        u_ref = data.get('usol', data.get('u', None))
        if u_ref is None:
            return None

        y_ref = None
        if 't' in data:
            y_ref = data['t'].flatten()
        elif 'y' in data:
            y_ref = data['y'].flatten()

        if u_ref.ndim == 2:
            ref_dict = {
                'x_ref': x_ref.flatten(),
                'u_ref': u_ref.flatten(),
                'file_path': mat_file
            }
            if y_ref is not None:
                ref_dict['y_ref'] = y_ref
                ref_dict['y_label'] = 't'
            return ref_dict
        else:
            return {
                'x_ref': x_ref.flatten(),
                'u_ref': u_ref.flatten(),
                'file_path': mat_file
            }

    def _get_spotter_net_class(self):
        from ..pseudotime_pde_solver.core.spotter_init import SpotterNet
        return SpotterNet
