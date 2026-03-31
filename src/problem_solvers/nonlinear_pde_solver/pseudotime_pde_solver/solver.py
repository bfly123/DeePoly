import os
import numpy as np
import time
from typing import Optional, Dict, Any, Tuple

import torch

from ..solver import NonlinearPDESolver
from .pseudotime_solver_factory_mixin import PseudoTimeSolverFactoryMixin
from .pseudotime_solver_iteration_mixin import PseudoTimeSolverIterationMixin


class PseudoTimePDESolver(
    PseudoTimeSolverFactoryMixin,
    PseudoTimeSolverIterationMixin,
    NonlinearPDESolver,
):
    DEFAULT_TIME_SCHEME = "implicit_euler"

    def _save_spotter_model_checkpoint(self, results_dir: str) -> Optional[str]:
        os.makedirs(results_dir, exist_ok=True)
        checkpoint_path = os.path.join(results_dir, "spotter_model.pth")
        try:
            torch.save(self.model.state_dict(), checkpoint_path)
        except Exception as exc:
            print(f"  Warning: failed to save Spotter model checkpoint: {exc}")
            return None
        print(f"  Spotter model saved: {checkpoint_path}")
        return checkpoint_path

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict[str, Any]:
        U = self._initialize_from_config() if U_init is None else np.asarray(U_init).copy()
        self.fitter.fitter_init(self._get_spotter_feature_source())

        start_time = time.time()
        dt = self.fitter.dt_init
        tau = 0.0
        coeffs = self._resolve_initial_coefficients()

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
            },
            "predictions_train": {
                'sniper': self._to_numpy_cpu(U_final),
                'spotter': self._to_numpy_cpu(U_spotter)
            },
            "flow_diagnostics": self._build_flow_diagnostic_results(coeffs_final),
        }

        self.visualizer.render(results, self.data_train, self.data_test, self.reference_solution)
        results["spotter_model_path"] = self._save_spotter_model_checkpoint(self.config.results_dir)
        return results

    def _resolve_initial_coefficients(self) -> np.ndarray:
        default_coeffs = self.fitter._initialize_coefficients()
        transferred = getattr(self, "_transferred_coeffs_init", None)
        if transferred is None:
            return default_coeffs

        coeffs = np.asarray(transferred, dtype=np.float64)
        if coeffs.shape != default_coeffs.shape:
            print(
                "  Warning: transferred coefficient shape mismatch "
                f"{coeffs.shape} != {default_coeffs.shape}; using zero coefficients"
            )
            return default_coeffs

        print(f"  Reusing transferred Snipper coefficients: shape={coeffs.shape}")
        return coeffs.copy()
