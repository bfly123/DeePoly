import os
from typing import Callable, Dict, Optional

import numpy as np


class LegacyAnalysisMixin:
    """Compatibility helpers for legacy case output modules."""

    def get_model_predictions(self, model, data: Dict) -> np.ndarray:
        torch = None
        try:
            import torch as _torch

            torch = _torch
        except ImportError:
            torch = None

        if torch is None:
            raise RuntimeError("Torch is required to evaluate model predictions")

        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            x = torch.as_tensor(data["x"], dtype=torch.float64)
            try:
                device = next(model.parameters()).device
                x = x.to(device)
            except StopIteration:
                pass
            with torch.no_grad():
                output = model(x)
            if isinstance(output, tuple):
                output = output[-1]
            return self._to_numpy_cpu(output)
        finally:
            if was_training:
                model.train()

    def get_deepoly_predictions(self, fitter, data: Dict, model, coeffs: np.ndarray) -> np.ndarray:
        prediction, _ = fitter.construct(data, model, coeffs)
        return np.asarray(prediction, dtype=np.float64)

    def calculate_errors(self, prediction: np.ndarray, exact: np.ndarray) -> Dict[str, float]:
        pred = np.asarray(prediction, dtype=np.float64)
        ref = np.asarray(exact, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        n_cols = min(pred.shape[1], ref.shape[1])
        pred = pred[:, :n_cols]
        ref = ref[:, :n_cols]
        abs_error = np.abs(pred - ref)
        return {
            "mse": float(np.mean(abs_error**2)),
            "rmse": float(np.sqrt(np.mean(abs_error**2))),
            "mae": float(np.mean(abs_error)),
            "max_error": float(np.max(abs_error)),
            "rel_error": float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12)),
        }

    def _print_error_statistics(self, **named_errors: Dict[str, float]) -> None:
        print("\n=== Error Statistics ===")
        print(f"{'Method':<18}{'MSE':<14}{'MAE':<14}{'Max Error':<14}{'Rel Error':<14}")
        print("-" * 74)
        for name, metrics in named_errors.items():
            if not isinstance(metrics, dict):
                continue
            print(
                f"{name:<18}"
                f"{metrics.get('mse', float('nan')):<14.6e}"
                f"{metrics.get('mae', float('nan')):<14.6e}"
                f"{metrics.get('max_error', float('nan')):<14.6e}"
                f"{metrics.get('rel_error', float('nan')):<14.6e}"
            )

    def _save_error_analysis_report(
        self,
        result_dir: str,
        errors: Dict[str, Dict[str, float]],
        timing_info: Optional[Dict[str, float]] = None,
    ) -> None:
        os.makedirs(result_dir, exist_ok=True)
        report_path = os.path.join(result_dir, "error_analysis_report.txt")
        with open(report_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("=" * 80 + "\n")
            file_obj.write("COMPREHENSIVE ERROR ANALYSIS REPORT\n")
            file_obj.write("=" * 80 + "\n\n")
            if timing_info:
                file_obj.write("TIMING INFORMATION:\n")
                file_obj.write("-" * 40 + "\n")
                file_obj.write(f"Total Solution Time: {timing_info.get('total_time', 0.0):.4f} seconds\n")
                file_obj.write(f"Neural Network Training Time (Scoper): {timing_info.get('scoper_time', 0.0):.4f} seconds\n")
                file_obj.write(f"Equation Fitting Time (Sniper): {timing_info.get('sniper_time', 0.0):.4f} seconds\n\n")

            file_obj.write("ERROR METRICS:\n")
            file_obj.write("-" * 40 + "\n")
            file_obj.write(
                f"{'Method':<18}{'MSE':<14}{'MAE':<14}{'Max Error':<14}{'Rel Error':<14}\n"
            )
            file_obj.write("-" * 74 + "\n")
            for name, metrics in errors.items():
                file_obj.write(
                    f"{name:<18}"
                    f"{metrics.get('mse', float('nan')):<14.6e}"
                    f"{metrics.get('mae', float('nan')):<14.6e}"
                    f"{metrics.get('max_error', float('nan')):<14.6e}"
                    f"{metrics.get('rel_error', float('nan')):<14.6e}\n"
                )

    def _resolve_exact_solution(self, exact_solution_func, x: np.ndarray) -> np.ndarray:
        if hasattr(self, "_resolve_reference_values"):
            resolved = self._resolve_reference_values(x, exact_solution_func)
        else:
            resolved = exact_solution_func(x)
        resolved = np.asarray(resolved, dtype=np.float64)
        if resolved.ndim == 1:
            resolved = resolved.reshape(-1, 1)
        return resolved

    def _save_basic_artifacts(self, result_dir: str, coeffs: np.ndarray, model) -> None:
        os.makedirs(result_dir, exist_ok=True)
        np.save(os.path.join(result_dir, "coefficients.npy"), coeffs)
        try:
            import torch

            torch.save(model.state_dict(), os.path.join(result_dir, "model.pt"))
        except Exception:
            pass

    def _run_common_analysis(
        self,
        *,
        data_train: Dict,
        data_test: Dict,
        model,
        fitter,
        coeffs: np.ndarray,
        exact_solution_func: Callable,
        result_dir: str,
        timing_info: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        net_train = self.get_model_predictions(model, data_train)
        net_test = self.get_model_predictions(model, data_test)
        deepoly_train = self.get_deepoly_predictions(fitter, data_train, model, coeffs)
        deepoly_test = self.get_deepoly_predictions(fitter, data_test, model, coeffs)
        exact_train = self._resolve_exact_solution(exact_solution_func, data_train["x"])
        exact_test = self._resolve_exact_solution(exact_solution_func, data_test["x"])

        errors = {
            "PINNs Train": self.calculate_errors(net_train, exact_train),
            "PINNs Test": self.calculate_errors(net_test, exact_test),
            "DeePoly Train": self.calculate_errors(deepoly_train, exact_train),
            "DeePoly Test": self.calculate_errors(deepoly_test, exact_test),
        }
        self._print_error_statistics(**errors)
        self._save_error_analysis_report(result_dir, errors, timing_info)
        self._save_basic_artifacts(result_dir, coeffs, model)
        return {
            "errors": errors,
            "net_train": net_train,
            "net_test": net_test,
            "deepoly_train": deepoly_train,
            "deepoly_test": deepoly_test,
            "exact_train": exact_train,
            "exact_test": exact_test,
        }

    def generate_1d_analysis(
        self,
        *,
        data_train: Dict,
        data_test: Dict,
        model,
        fitter,
        coeffs: np.ndarray,
        exact_solution_func: Callable,
        result_dir: str,
        variables=None,
        timing_info: Optional[Dict[str, float]] = None,
        **_kwargs,
    ) -> None:
        _ = variables
        analysis = self._run_common_analysis(
            data_train=data_train,
            data_test=data_test,
            model=model,
            fitter=fitter,
            coeffs=coeffs,
            exact_solution_func=exact_solution_func,
            result_dir=result_dir,
            timing_info=timing_info,
        )
        vis_dir = os.path.join(result_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        self.plot_solution_1d(
            data_test["x"],
            analysis["exact_test"],
            analysis["net_test"],
            save_path=os.path.join(vis_dir, "pinn_test.png"),
            title="PINNs Test Solution",
        )
        self.plot_solution_1d(
            data_test["x"],
            analysis["exact_test"],
            analysis["deepoly_test"],
            save_path=os.path.join(vis_dir, "deepoly_test.png"),
            title="DeePoly Test Solution",
        )
        if hasattr(self, "plot_reference_comparison"):
            self.plot_reference_comparison(
                data_test["x"],
                analysis["deepoly_test"],
                exact_solution_func,
                save_path=os.path.join(vis_dir, "deepoly_reference_comparison.png"),
                title="DeePoly vs Reference",
            )
        if hasattr(self, "plot_error_distribution"):
            self.plot_error_distribution(
                analysis["deepoly_test"],
                analysis["exact_test"],
                save_path=os.path.join(vis_dir, "deepoly_error_distribution.png"),
                title="DeePoly Error Distribution",
            )
        if hasattr(self, "compute_spatial_error_stats") and hasattr(self, "plot_error_statistics"):
            spatial_stats = self.compute_spatial_error_stats(
                data_test["x"],
                np.abs(analysis["deepoly_test"] - analysis["exact_test"]),
            )
            self.plot_error_statistics(
                spatial_stats,
                save_path=os.path.join(vis_dir, "deepoly_spatial_error_stats.png"),
                title="DeePoly Spatial Error Statistics",
            )

    def generate_2d_analysis(
        self,
        *,
        data_train: Dict,
        data_test: Dict,
        model,
        fitter,
        coeffs: np.ndarray,
        exact_solution_func: Callable,
        result_dir: str,
        variables=None,
        timing_info: Optional[Dict[str, float]] = None,
        **_kwargs,
    ) -> None:
        _ = variables
        analysis = self._run_common_analysis(
            data_train=data_train,
            data_test=data_test,
            model=model,
            fitter=fitter,
            coeffs=coeffs,
            exact_solution_func=exact_solution_func,
            result_dir=result_dir,
            timing_info=timing_info,
        )
        vis_dir = os.path.join(result_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        self.plot_solution_2d(
            data_test["x"],
            analysis["exact_test"],
            analysis["net_test"],
            save_path=os.path.join(vis_dir, "pinn_test.png"),
            title="PINNs Test Solution",
        )
        self.plot_solution_2d(
            data_test["x"],
            analysis["exact_test"],
            analysis["deepoly_test"],
            save_path=os.path.join(vis_dir, "deepoly_test.png"),
            title="DeePoly Test Solution",
        )
        if hasattr(self, "plot_reference_comparison"):
            self.plot_reference_comparison(
                data_test["x"],
                analysis["deepoly_test"],
                exact_solution_func,
                save_path=os.path.join(vis_dir, "deepoly_reference_comparison.png"),
                title="DeePoly vs Reference",
            )
        if hasattr(self, "plot_error_field_2d"):
            self.plot_error_field_2d(
                data_test["x"],
                analysis["deepoly_test"],
                analysis["exact_test"],
                save_path=os.path.join(vis_dir, "deepoly_error_field.png"),
                title="DeePoly Error Field",
            )

    def generate_basic_analysis(
        self,
        *,
        data_train: Dict,
        data_test: Dict,
        model,
        fitter,
        coeffs: np.ndarray,
        exact_solution_func: Callable,
        result_dir: str,
        timing_info: Optional[Dict[str, float]] = None,
        variables=None,
        **_kwargs,
    ) -> None:
        _ = variables
        self._run_common_analysis(
            data_train=data_train,
            data_test=data_test,
            model=model,
            fitter=fitter,
            coeffs=coeffs,
            exact_solution_func=exact_solution_func,
            result_dir=result_dir,
            timing_info=timing_info,
        )

    def plot_2d_comparison(self, **kwargs) -> None:
        n_dim = int(getattr(self, "n_dim", 1) or 1)
        if n_dim == 1:
            self.generate_1d_analysis(**kwargs)
            return
        if n_dim == 2:
            self.generate_2d_analysis(**kwargs)
            return
        self.generate_basic_analysis(**kwargs)
