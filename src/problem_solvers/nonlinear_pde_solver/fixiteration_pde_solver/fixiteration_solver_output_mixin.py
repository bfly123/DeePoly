import json
import os
from typing import Any, Dict

import numpy as np

from src.abstract_class.utils.reference_solution_handler import ReferenceSolutionHandler


class FixIterationSolverOutputMixin:
    def _postprocess_results(self, U_final, coeffs_final, conv_info):
        _ = coeffs_final
        final_stage = conv_info.get("final_stage", "snipper" if coeffs_final is not None else "spotter")
        U_network_test = self._predict_field_on_points(self.data_test["x"])
        precomputed_test = conv_info.get("final_test_solution", None)
        if precomputed_test is not None:
            U_test = self._to_numpy_cpu(precomputed_test)
        elif final_stage == "spotter":
            U_test = U_network_test
        else:
            U_test = self._construct_test_field_from_coeffs(coeffs_final, U_final)

        results_dir = self.config.results_dir
        self.visualizer.save_convergence_history(conv_info, results_dir)
        self.visualizer.plot_final_solution(
            U_test,
            self.data_test,
            os.path.join(results_dir, "final_solution.png"),
        )
        residual_history = conv_info.get("residual_history", [])
        self.visualizer.plot_residual_history(
            residual_history,
            list(range(1, len(residual_history) + 1)),
            os.path.join(results_dir, "residual_history.png")
        )
        coeffs_to_save = coeffs_final if coeffs_final is not None else np.empty((0,), dtype=np.float64)
        self.visualizer.save_solution_data(
            U_train=U_final,
            U_spotter_test=U_network_test,
            U_test=U_test,
            coeffs_steady=coeffs_to_save,
            conv_info=conv_info,
            data_train=self._get_iteration_dataset(),
            data_test=self.data_test,
            save_dir=results_dir,
        )
        self._save_accuracy_report(
            U_network_test=U_network_test,
            U_test=U_test,
            conv_info=conv_info,
            coeffs_final=coeffs_final,
            results_dir=results_dir,
        )
        self._save_final_comparison(U_test, final_stage)

    def _save_network_comparison(self):
        if not hasattr(self, "reference_solution") or self.reference_solution is None:
            return
        U_network_test = self._predict_field_on_points(self.data_test["x"])
        save_path = os.path.join(self.config.results_dir, "network_vs_reference.png")
        self.visualizer.plot_reference_comparison(
            self.data_test["x"],
            U_network_test,
            self.reference_solution,
            save_path,
        )

    def _save_final_comparison(self, U_test, final_stage: str):
        if not hasattr(self, "reference_solution") or self.reference_solution is None:
            return
        filename = "sniper_vs_reference.png" if final_stage == "snipper" else "bc_pinn_vs_reference.png"
        save_path = os.path.join(self.config.results_dir, filename)
        self.visualizer.plot_reference_comparison(
            self.data_test["x"],
            U_test,
            self.reference_solution,
            save_path,
        )

    def _construct_test_field_from_coeffs(self, coeffs_final, U_final):
        if coeffs_final is not None:
            try:
                U_test, _ = self.fitter.construct(
                    self.data_test,
                    self._get_spotter_feature_source(),
                    coeffs_final,
                )
                return self._to_numpy_cpu(U_test)
            except Exception as exc:
                print(f"  Warning: direct Snipper reconstruction on test grid failed: {exc}")

        U_final_cpu = self._to_numpy_cpu(U_final)
        return self._interpolate_to_test_grid(
            U_final_cpu,
            data_source=self._get_iteration_dataset(),
        )

    def get_solver_time_info(self) -> Dict[str, Any]:
        info = super().get_solver_time_info()
        info.update(
            {
                "solver_type": "fixed_point_iteration",
                "has_time_stepping": False,
                "convergence_based": True,
                "max_iterations": getattr(self.fitter, "max_iterations", "N/A"),
                "convergence_tolerance": getattr(self.fitter, "convergence_tol", "N/A"),
            }
        )
        return info

    def _save_accuracy_report(
        self,
        U_network_test,
        U_test,
        conv_info: Dict[str, Any],
        coeffs_final,
        results_dir: str,
    ) -> None:
        if not hasattr(self, "reference_solution") or self.reference_solution is None:
            return

        network_metrics = self._compute_reference_metrics(self.data_test["x"], U_network_test)
        final_metrics = self._compute_reference_metrics(self.data_test["x"], U_test)
        final_time_metrics = self._compute_final_slice_metrics(self.data_test["x"], U_network_test, U_test)
        poly_degree = self._normalize_int_sequence(getattr(self.config, "poly_degree", None))

        report = {
            "spotter": network_metrics,
            "snipper_direct": final_metrics,
            "final_time_rel_l2": final_time_metrics,
            "fixed_point": {
                "iterations": int(conv_info.get("iterations", 0)),
                "final_residual": float(conv_info.get("final_residual", 0.0)),
            },
            "snipper_system": {
                "segments": list(getattr(self.config, "n_segments", [])),
                "poly_degree": poly_degree,
                "poly_basis_dim": int(np.prod(np.asarray(poly_degree, dtype=int) + 1)) if poly_degree else None,
                "nn_feature_dim": int(getattr(self.fitter, "nn_feature_dim", 0)),
                "domain_points": int(getattr(self.config, "points_domain", 0)),
                "boundary_points": self._extract_boundary_point_count(),
                "points_per_swap": int(getattr(self.config, "points_per_swap", 0)),
                "linear_method": getattr(self.config, "linear_method", None),
                "feature_dtype": self._extract_feature_dtype(),
                "under_relaxation": float(getattr(self.config, "under_relaxation", 1.0)),
            },
        }

        path = os.path.join(results_dir, "accuracy_report.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def _compute_reference_metrics(self, coords, values) -> Dict[str, float]:
        numerical_coords = np.asarray(coords, dtype=np.float64)
        numerical_values = np.asarray(values, dtype=np.float64).reshape(-1)
        payload = ReferenceSolutionHandler.prepare_comparison(
            numerical_coords,
            numerical_values,
            self.reference_solution,
        )
        reference_values = ReferenceSolutionHandler.interpolate_to_points(
            payload.reference_coords,
            payload.reference_values,
            payload.numerical_coords,
            method=payload.interpolation_method,
        )
        error = numerical_values - np.asarray(reference_values, dtype=np.float64).reshape(-1)
        ref_norm = float(np.linalg.norm(reference_values)) + 1e-12
        return {
            "rel_l2": float(np.linalg.norm(error) / ref_norm),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "max_abs": float(np.max(np.abs(error))),
        }

    def _compute_final_slice_metrics(self, coords, network_values, final_values) -> Dict[str, float]:
        numerical_coords = np.asarray(coords, dtype=np.float64)
        if numerical_coords.ndim != 2 or numerical_coords.shape[1] != 2:
            return {}

        target_axis_value = float(np.max(numerical_coords[:, 1]))
        mask = np.isclose(numerical_coords[:, 1], target_axis_value)
        if not np.any(mask):
            return {}

        final_coords = numerical_coords[mask]
        payload = ReferenceSolutionHandler.prepare_comparison(
            final_coords,
            np.asarray(final_values, dtype=np.float64).reshape(-1)[mask],
            self.reference_solution,
        )
        reference_values = ReferenceSolutionHandler.interpolate_to_points(
            payload.reference_coords,
            payload.reference_values,
            payload.numerical_coords,
            method=payload.interpolation_method,
        )
        ref_norm = float(np.linalg.norm(reference_values)) + 1e-12

        spotter_err = np.asarray(network_values, dtype=np.float64).reshape(-1)[mask] - np.asarray(reference_values, dtype=np.float64).reshape(-1)
        snipper_err = np.asarray(final_values, dtype=np.float64).reshape(-1)[mask] - np.asarray(reference_values, dtype=np.float64).reshape(-1)
        return {
            "spotter": float(np.linalg.norm(spotter_err) / ref_norm),
            "snipper_direct": float(np.linalg.norm(snipper_err) / ref_norm),
        }

    def _extract_boundary_point_count(self) -> int:
        boundary_conditions = getattr(self.config, "boundary_conditions", []) or []
        total = 0
        for item in boundary_conditions:
            if not isinstance(item, dict):
                continue
            total += int(item.get("points", 0) or 0)
        return total

    def _extract_feature_dtype(self):
        feature_cfg = getattr(self.config, "spotter_feature", {}) or {}
        if not isinstance(feature_cfg, dict):
            return None
        options = feature_cfg.get("options")
        if isinstance(options, dict) and options.get("dtype") is not None:
            return options.get("dtype")
        factory_kwargs = feature_cfg.get("factory_kwargs", {}) or {}
        if not isinstance(factory_kwargs, dict):
            return None
        return factory_kwargs.get("dtype")

    @staticmethod
    def _normalize_int_sequence(values) -> list[int]:
        if values is None:
            return []
        array = np.asarray(values, dtype=int).reshape(-1)
        return [int(item) for item in array.tolist()]
