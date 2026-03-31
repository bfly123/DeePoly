import os

import matplotlib.pyplot as plt
import numpy as np


class DimensionalPlotterReferenceMixin:
    def plot_comparison_with_reference(
        self,
        U_numerical,
        data,
        reference_solution,
        save_path: str,
        **kwargs,
    ) -> None:
        if reference_solution is None:
            self.plot_solution(U_numerical, data, save_path, **kwargs)
            return
        if self.n_dim == 1:
            self._plot_1d_comparison(U_numerical, data, reference_solution, save_path, **kwargs)
            return
        if self.n_dim == 2:
            self._plot_2d_comparison(U_numerical, data, reference_solution, save_path, **kwargs)
            return
        raise ValueError(f"Unsupported dimension: {self.n_dim}")

    def _plot_1d_comparison(self, U_numerical, data, reference_solution, save_path: str, **kwargs) -> None:
        x_num = data["x"].flatten()
        x_ref = reference_solution["x_ref"].flatten()
        u_num = U_numerical.flatten() if len(U_numerical.shape) > 1 else U_numerical
        u_ref = reference_solution["u_ref"].flatten()
        plt.figure(figsize=kwargs.get("figsize", (12, 6)))
        plt.subplot(1, 2, 1)
        plt.plot(x_num, u_num, "b-", label="Numerical", linewidth=2)
        plt.plot(x_ref, u_ref, "r--", label="Reference", linewidth=2)
        plt.xlabel(self.spatial_vars[0])
        plt.ylabel("u")
        plt.title("Solution Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        x_err, error = self._interpolated_1d_error(x_num, u_num, x_ref, u_ref)
        plt.plot(x_err, error, "g-", linewidth=2)
        plt.xlabel(self.spatial_vars[0])
        plt.ylabel("Absolute Error")
        plt.title("Error Analysis")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=kwargs.get("dpi", 100), bbox_inches="tight")
        plt.close()

    def _interpolated_1d_error(self, x_num, u_num, x_ref, u_ref):
        from scipy.interpolate import interp1d

        if len(x_ref) > len(x_num):
            ref_interp = interp1d(x_ref, u_ref, kind="linear", fill_value="extrapolate")
            return x_num, np.abs(u_num - ref_interp(x_num))
        num_interp = interp1d(x_num, u_num, kind="linear", fill_value="extrapolate")
        return x_ref, np.abs(num_interp(x_ref) - u_ref)

    def _plot_2d_comparison(self, U_numerical, data, reference_solution, save_path: str, **kwargs) -> None:
        ref_solution_sliced = self._prepare_2d_reference_slice(reference_solution, kwargs.get("time_slice", 0.5))
        self._plot_2d_solution(U_numerical, data, save_path.replace(".png", "_numerical.png"), **kwargs)
        if "u_ref" not in ref_solution_sliced:
            return
        ref_data = {"x": ref_solution_sliced["x_ref"]}
        self._plot_2d_solution(ref_solution_sliced["u_ref"], ref_data, save_path.replace(".png", "_reference.png"), **kwargs)
        grid_shape = kwargs.get("grid_shape", (100, 100))
        if self._can_compare_2d_grids(data, ref_solution_sliced, grid_shape):
            error_data = self._compute_2d_error(U_numerical, data, ref_solution_sliced["u_ref"], ref_solution_sliced, grid_shape)
            self._plot_2d_solution(
                error_data["error"],
                error_data,
                save_path.replace(".png", "_error.png"),
                **{**kwargs, "colormap": "viridis"},
            )

    def _prepare_2d_reference_slice(self, reference_solution, time_value: float):
        if "x_ref" in reference_solution and len(reference_solution["x_ref"].shape) == 2:
            if reference_solution["x_ref"].shape[1] == 3:
                return self._slice_2d_to_2d(reference_solution, time_value)
        return reference_solution

    def _slice_2d_to_2d(self, ref_solution, time_value: float):
        x_ref = ref_solution["x_ref"]
        u_ref = ref_solution["u_ref"]
        if len(x_ref.shape) == 3 and x_ref.shape[2] > 1:
            time_coords = x_ref[0, 0, :] if len(u_ref.shape) == 3 else x_ref[0, :, -1]
            time_idx = np.argmin(np.abs(time_coords - time_value))
            u_slice = u_ref[:, :, time_idx] if len(u_ref.shape) == 3 else u_ref[:, time_idx]
            coords = x_ref[:, :, : time_idx + 1].reshape(-1, 2) if time_idx > 0 else x_ref[:, :, 0].reshape(-1, 2)
            return {"x_ref": coords, "u_ref": u_slice.flatten() if len(u_slice.shape) > 1 else u_slice}
        return ref_solution

    @staticmethod
    def _can_compare_2d_grids(data_num, ref_solution, _grid_shape) -> bool:
        try:
            return data_num["x"].shape[0] == ref_solution["x_ref"].shape[0]
        except Exception:
            return False

    @staticmethod
    def _compute_2d_error(U_numerical, data_num, u_ref, _ref_solution, grid_shape):
        u_num_reshaped = U_numerical.reshape(grid_shape)
        u_ref_reshaped = u_ref.reshape(grid_shape)
        return {"x": data_num["x"], "error": np.abs(u_num_reshaped - u_ref_reshaped).flatten()}

    def get_dimension_info(self):
        return {
            "n_dim": self.n_dim,
            "spatial_vars": self.spatial_vars,
            "plot_type": "1D line plot" if self.n_dim == 1 else "2D contour plot",
        }
