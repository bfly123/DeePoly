import json
import os
from typing import Dict, Optional

import numpy as np


class PseudotimeVisualizeOutputMixin:
    def plot_solution(
        self,
        data: Dict,
        prediction: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """Retain the single-shot plotting API for external callers."""
        if save_path:
            save_dir = os.path.dirname(save_path) or "."
            output_name = os.path.splitext(os.path.basename(save_path))[0]
        else:
            save_dir = getattr(self.config, "results_dir", ".")
            output_name = "solution_preview"
            save_path = os.path.join(save_dir, f"{output_name}.png")

        prepared = self._prepare_plot_data(data)
        if prepared is None:
            print("  Warning: Unable to plot solution preview due to missing coordinates.")
            return

        values = self._to_numpy_cpu(prediction)
        if values is None:
            return

        plot_kwargs = self._build_plot_kwargs(dataset="test")
        self.dimensional_visualizer.plot_solution(values, prepared, save_path, **plot_kwargs)

    def _save_convergence_plots(self, conv_info, save_dir):
        if "residual_history" not in conv_info or not conv_info["residual_history"]:
            return
        residual_history = conv_info["residual_history"]
        iterations = list(range(len(residual_history)))
        tau_history = conv_info.get("tau_history", None)
        self.plot_residual_history(
            residual_history,
            iterations,
            os.path.join(save_dir, "convergence_history.png"),
            tau_history,
        )

    def _save_data(self, results, data_train, data_test, save_dir):
        save_data = {
            "coeffs": self._to_numpy_cpu(results["coefficients"]),
            "converged": results["convergence_info"]["converged"],
        }
        prepared_test = self._prepare_plot_data(data_test)
        if prepared_test and "x" in prepared_test:
            save_data["x_test"] = prepared_test["x"]
        prepared_train = self._prepare_plot_data(data_train)
        if prepared_train and "x" in prepared_train:
            save_data["x_train"] = prepared_train["x"]

        for output_name, U_pred in results["predictions"].items():
            save_data[f"{output_name}_test"] = self._to_numpy_cpu(U_pred)
        for output_name, U_pred in results.get("predictions_train", {}).items():
            save_data[f"{output_name}_train"] = self._to_numpy_cpu(U_pred)

        np.savez(os.path.join(save_dir, "solution_data.npz"), **save_data)
        print(f"  Data saved: {os.path.join(save_dir, 'solution_data.npz')}")
        convergence_path = os.path.join(save_dir, "convergence_info.json")
        with open(convergence_path, "w", encoding="utf-8") as handle:
            json.dump(
                self._to_jsonable(results["convergence_info"]),
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  Convergence info saved: {convergence_path}")

    def _to_jsonable(self, value):
        if isinstance(value, dict):
            return {key: self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _should_plot_reference(self):
        policy = self.config.visualization.reference_policy
        policy_value = getattr(policy, "value", policy)
        return policy_value in ["auto", "force"]
