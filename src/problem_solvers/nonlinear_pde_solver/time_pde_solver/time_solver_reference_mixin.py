from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


class TimeSolverReferenceMixin:
    def get_reference_solution_at_time(self, t: float) -> Optional[np.ndarray]:
        """Get reference solution at specified time"""
        if self.reference_solution is None:
            return None
        try:
            x_ref = self.reference_solution["x_ref"]
            interpolators = self.reference_solution["interpolators"]
            u_at_t = np.zeros(len(x_ref))
            for index in range(len(x_ref)):
                u_at_t[index] = interpolators[index](t)
            return u_at_t
        except Exception as exc:
            print(f"Warning: Cannot get reference solution at time t={t}: {exc}")
            return None

    def _process_reference_solution(self, data: Dict, x_ref: np.ndarray, mat_file: str) -> Optional[Dict]:
        """Process reference solution data to match time solver expectations"""
        if "t" not in data:
            print("Warning: 't' array not found in reference solution file")
            return None
        t_ref = data["t"].flatten()
        u_ref = self._extract_reference_solution_array(data)
        if u_ref is None:
            return None
        u_ref = self._normalize_reference_solution_shape(np.asarray(u_ref), x_ref, t_ref)
        if u_ref is None:
            return None

        print(f"  Time points: {len(t_ref)}")
        print(f"  Spatial points: {len(x_ref)}")
        print(f"  Time range: [{t_ref.min():.6f}, {t_ref.max():.6f}]")
        print(f"  Spatial range: [{x_ref.min():.6f}, {x_ref.max():.6f}]")

        interpolators = {}
        for index in range(len(x_ref)):
            interpolators[index] = interp1d(
                t_ref,
                u_ref[:, index],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )

        return {
            "t_ref": t_ref,
            "x_ref": x_ref,
            "u_ref": u_ref,
            "interpolators": interpolators,
            "file_path": mat_file,
        }

    def _extract_reference_solution_array(self, data: Dict):
        if "usol" in data:
            return data["usol"]
        if "u" in data:
            return data["u"]
        print("Warning: Solution field 'usol' not found in reference solution file")
        return None

    def _normalize_reference_solution_shape(self, u_ref: np.ndarray, x_ref: np.ndarray, t_ref: np.ndarray):
        if u_ref.ndim == 2:
            if u_ref.shape[0] == len(x_ref) and u_ref.shape[1] == len(t_ref):
                return u_ref.T
            if u_ref.shape == (len(t_ref), len(x_ref)):
                return u_ref
            print("  Warning: Unexpected reference solution shape, attempting to reshape")
            return u_ref.reshape(len(t_ref), -1)
        if u_ref.ndim > 2:
            return u_ref.reshape(len(t_ref), -1)
        return u_ref

    def get_solution_at_time(self, T_target: float) -> Tuple[np.ndarray, float]:
        """Get指定Time的Solution"""
        time_history, solution_history = self.visualizer.get_animation_data()
        if not time_history:
            raise RuntimeError("Need先Runningsolve()method")
        time_array = np.array(time_history)
        idx = np.argmin(np.abs(time_array - T_target))
        return solution_history[idx], time_array[idx]

    def get_convergence_info(self) -> Dict:
        """GetConvergentinformation"""
        if not self.loss_history:
            return {"error": "没有Available的ConvergentData"}
        return {
            "final_loss": self.loss_history[-1],
            "initial_loss": self.loss_history[0],
            "reduction_ratio": self.loss_history[0] / self.loss_history[-1],
            "total_steps": len(self.loss_history),
            "average_loss": np.mean(self.loss_history),
        }

    def export_solution_data(self, filepath: str):
        """ExportSolutionData"""
        time_history, solution_history = self.visualizer.get_animation_data()
        export_data = {
            "config": self.config.__dict__,
            "time_history": time_history,
            "solution_history": solution_history,
            "loss_history": self.loss_history,
            "x_coords": self.data_test["x"].flatten() if "x" in self.data_test else np.array([]),
        }
        np.savez_compressed(filepath, **export_data)
        print(f"SolutionDataExport至: {filepath}")
