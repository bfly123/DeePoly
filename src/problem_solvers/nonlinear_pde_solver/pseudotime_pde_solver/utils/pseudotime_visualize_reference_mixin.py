import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class PseudoTimeReferenceMixin:
    def _save_reference_comparison(
        self,
        output_name: str,
        coords: np.ndarray,
        prediction: np.ndarray,
        reference_solution: Dict,
        results_dir: str,
    ) -> None:
        u_pred = self._to_numpy_cpu(prediction)
        if u_pred is None:
            return

        save_path = os.path.join(results_dir, f"{output_name}_reference_comparison.png")
        if self.n_dim == 1:
            ref_callable = self._prepare_reference_callable(reference_solution)
            title = f"{output_name.capitalize()} Reference Comparison"
            self.plot_reference_comparison(coords, u_pred, ref_callable, save_path=save_path, title=title)
            return
        if self.n_dim == 2:
            self._plot_reference_comparison_2d(coords, u_pred, reference_solution, results_dir, output_name)
            return
        print(f"  Warning: Reference comparison not implemented for {self.n_dim}D problems.")

    def _prepare_reference_callable(self, reference_solution):
        if callable(reference_solution):
            return reference_solution

        if isinstance(reference_solution, dict) and self.n_dim == 1 and {"x_ref", "u_ref"} <= reference_solution.keys():
            x_ref = np.asarray(reference_solution["x_ref"]).reshape(-1)
            u_ref = np.asarray(reference_solution["u_ref"]).reshape(-1)

            def ref_func(x: np.ndarray):
                x_flat = x[:, 0] if x.ndim > 1 else x
                values = np.interp(x_flat, x_ref, u_ref)
                return values.reshape(-1, 1)

            return ref_func

        return reference_solution

    def _plot_reference_comparison_2d(
        self,
        x_coords: np.ndarray,
        u_pred: np.ndarray,
        reference_solution: Dict,
        results_dir: str,
        output_name: str,
    ) -> None:
        x_flat = x_coords[:, 0]
        y_flat = x_coords[:, 1]
        x_unique = np.unique(x_flat)
        y_unique = np.unique(y_flat)

        U_pred_grid = self._map_to_grid(x_flat, y_flat, u_pred.flatten(), x_unique, y_unique)
        ref_interp = self._interpolate_reference_to_grid(reference_solution, x_unique, y_unique)
        if ref_interp is None:
            print(f"  Warning: Unable to build 2D reference comparison for {output_name}, skipping")
            return

        X_mesh, Y_mesh, U_ref_grid = ref_interp
        abs_error = np.abs(U_pred_grid - U_ref_grid)
        colormap = getattr(self.config.visualization, "colormap", "RdBu_r")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].pcolormesh(X_mesh, Y_mesh, U_pred_grid, cmap=colormap, shading="gouraud")
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title(f"{output_name.capitalize()} Prediction")
        axes[0].set_xlabel(self.config.spatial_vars[0])
        axes[0].set_ylabel(self.config.spatial_vars[1])
        axes[0].set_aspect("equal")

        im1 = axes[1].pcolormesh(X_mesh, Y_mesh, U_ref_grid, cmap=colormap, shading="gouraud")
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title("Reference")
        axes[1].set_xlabel(self.config.spatial_vars[0])
        axes[1].set_ylabel(self.config.spatial_vars[1])
        axes[1].set_aspect("equal")

        im2 = axes[2].pcolormesh(X_mesh, Y_mesh, abs_error, cmap="viridis", shading="gouraud")
        fig.colorbar(im2, ax=axes[2])
        axes[2].set_title("|Prediction - Reference|")
        axes[2].set_xlabel(self.config.spatial_vars[0])
        axes[2].set_ylabel(self.config.spatial_vars[1])
        axes[2].set_aspect("equal")

        plt.tight_layout()
        save_path = os.path.join(results_dir, f"{output_name}_reference_comparison.png")
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _reshape_reference_grid(self, x_ref: np.ndarray, y_ref: np.ndarray, u_ref: np.ndarray):
        if u_ref.ndim == 1:
            try:
                return u_ref.reshape(len(x_ref), len(y_ref), order="C")
            except ValueError:
                try:
                    return u_ref.reshape(len(y_ref), len(x_ref), order="C").T
                except ValueError:
                    return None
        if u_ref.shape == (len(x_ref), len(y_ref)):
            return u_ref
        if u_ref.shape == (len(y_ref), len(x_ref)):
            return u_ref.T
        return None

    def _interpolate_reference_to_grid(
        self,
        reference_solution: Dict,
        x_unique: np.ndarray,
        y_unique: np.ndarray,
    ):
        if not isinstance(reference_solution, dict):
            print("  Warning: Reference solution is not a dict; cannot interpolate for 2D comparison.")
            return None

        required_keys = {"x_ref", "y_ref", "u_ref"}
        if not required_keys <= reference_solution.keys():
            print(f"  Warning: Reference solution missing keys {required_keys}, got {reference_solution.keys()}")
            return None

        x_ref = np.asarray(reference_solution["x_ref"]).reshape(-1)
        y_ref = np.asarray(reference_solution["y_ref"]).reshape(-1)
        u_ref = np.asarray(reference_solution["u_ref"])
        u_ref_grid = self._reshape_reference_grid(x_ref, y_ref, u_ref)
        if u_ref_grid is None:
            print(f"  Warning: Unexpected reference solution shape {u_ref.shape}")
            return None

        interpolator = RegularGridInterpolator((x_ref, y_ref), u_ref_grid, bounds_error=False, fill_value=None)
        X_mesh, Y_mesh = np.meshgrid(x_unique, y_unique, indexing="xy")
        points = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
        ref_values = interpolator(points)
        return X_mesh, Y_mesh, ref_values.reshape(len(y_unique), len(x_unique))
