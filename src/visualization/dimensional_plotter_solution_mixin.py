import os

import matplotlib.pyplot as plt
import numpy as np


class DimensionalPlotterSolutionMixin:
    def plot_solution(self, U, data, save_path: str, **kwargs) -> None:
        save_path = self._resolve_save_path(save_path)
        if self.n_dim == 1:
            self._plot_1d_solution(U, data, save_path, **kwargs)
            return
        if self.n_dim == 2:
            self._plot_2d_solution(U, data, save_path, **kwargs)
            return
        raise ValueError(f"Unsupported dimension: {self.n_dim}")

    def _resolve_save_path(self, save_path: str) -> str:
        if not os.path.isabs(save_path) and hasattr(self.config, "results_dir"):
            return os.path.join(self.config.results_dir, os.path.basename(save_path))
        return save_path

    def _plot_1d_solution(self, U, data, save_path: str, **kwargs) -> None:
        x = data["x"].flatten()
        n_eqs = U.shape[1] if len(U.shape) > 1 else 1
        kwargs.pop("plot_mode", None)
        kwargs.pop("scatter_kwargs", None)
        kwargs.pop("grid_kwargs", None)
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))
        if n_eqs == 1:
            self._plot_single_1d_series(x, U)
        else:
            self._plot_multi_1d_series(x, U, n_eqs)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=kwargs.get("dpi", 100), bbox_inches="tight")
        plt.close()

    def _plot_single_1d_series(self, x, U) -> None:
        y = U.flatten() if len(U.shape) > 1 else U
        plt.plot(x, y, linewidth=2)
        plt.xlabel(self.spatial_vars[0])
        plt.ylabel("u")
        plt.title("1D Solution")

    def _plot_multi_1d_series(self, x, U, n_eqs: int) -> None:
        for index in range(n_eqs):
            plt.plot(x, U[:, index].flatten(), label=f"u_{index}", linewidth=2)
        plt.xlabel(self.spatial_vars[0])
        plt.ylabel("u")
        plt.title("1D Solution (Multiple Variables)")
        plt.legend()

    def _plot_2d_solution(self, U, data, save_path: str, **kwargs) -> None:
        x = data["x"]
        n_eqs = U.shape[1] if len(U.shape) > 1 else 1
        if x.shape[1] != 2:
            raise ValueError("2D plotting requires exactly 2 spatial coordinates")
        plot_mode, scatter_kwargs, grid_settings, colormap = self._prepare_2d_plot_settings(kwargs)
        x_coords, y_coords = x[:, 0], x[:, 1]
        for eq_idx in range(n_eqs):
            self._plot_single_2d_field(
                U.flatten() if n_eqs == 1 else U[:, eq_idx],
                x_coords,
                y_coords,
                save_path if n_eqs == 1 else save_path.replace(".png", f"_eq{eq_idx}.png"),
                eq_idx,
                n_eqs,
                plot_mode,
                scatter_kwargs,
                grid_settings,
                colormap,
                kwargs,
            )

    def _prepare_2d_plot_settings(self, kwargs):
        plot_mode = kwargs.pop("plot_mode", None)
        scatter_kwargs = dict(kwargs.pop("scatter_kwargs", {}) or {})
        grid_kwargs = dict(kwargs.pop("grid_kwargs", {}) or {})
        vis_cfg = getattr(self.config, "visualization", None)
        if plot_mode is None and vis_cfg is not None:
            plot_mode = getattr(vis_cfg, "plot_mode_test", "auto")
        plot_mode = (plot_mode or "auto").lower()
        colormap = kwargs.get("colormap", getattr(vis_cfg, "colormap", "RdBu_r"))
        scatter_kwargs.setdefault("s", 18)
        scatter_kwargs.setdefault("cmap", colormap)
        scatter_kwargs.setdefault("edgecolors", "none")
        scatter_kwargs.setdefault("alpha", 0.85)
        settings = {
            "method": grid_kwargs.pop("method", kwargs.get("grid_method", "linear")),
            "shading": grid_kwargs.pop("shading", "auto"),
            "colorbar_kwargs": grid_kwargs.pop("colorbar_kwargs", {}),
            "show_colorbar": grid_kwargs.pop("colorbar", True),
            "unused_grid_kwargs": grid_kwargs,
        }
        return plot_mode, scatter_kwargs, settings, colormap

    @staticmethod
    def _is_structured_grid(x_coords, y_coords, point_count: int) -> bool:
        x_unique = np.unique(x_coords)
        y_unique = np.unique(y_coords)
        return len(x_unique) * len(y_unique) == point_count

    def _plot_single_2d_field(
        self,
        u_values,
        x_coords,
        y_coords,
        save_path,
        eq_idx: int,
        n_eqs: int,
        plot_mode,
        scatter_kwargs,
        grid_settings,
        colormap,
        kwargs,
    ) -> None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 8)))
        if grid_settings["unused_grid_kwargs"]:
            unknown_keys = ", ".join(grid_settings["unused_grid_kwargs"].keys())
            print(f"  Warning: Unused grid_kwargs keys ignored: {unknown_keys}")
        grid_payload = self._build_structured_grid_payload(x_coords, y_coords, u_values)
        use_grid = plot_mode in ("auto", "grid") and grid_payload is not None and plot_mode != "scatter"
        if use_grid:
            rendered = self._render_2d_grid(ax, grid_payload, colormap, grid_settings)
        else:
            if plot_mode == "grid" and grid_payload is None:
                print("  Warning: Requested grid plot but points are unstructured; using scatter instead.")
            rendered = self._render_2d_scatter(ax, x_coords, y_coords, u_values, scatter_kwargs)
        if grid_settings["show_colorbar"]:
            plt.colorbar(rendered, ax=ax, label="u", **grid_settings["colorbar_kwargs"])
        self._finalize_2d_axes(ax, eq_idx, n_eqs, kwargs)
        plt.tight_layout()
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=kwargs.get("dpi", 100), bbox_inches="tight")
        plt.close()

    def _render_2d_grid(self, ax, grid_payload, colormap, grid_settings):
        x_mesh, y_mesh, grid_z = grid_payload
        return ax.pcolormesh(x_mesh, y_mesh, grid_z, cmap=colormap, shading=grid_settings["shading"])

    def _build_structured_grid_payload(self, x_coords, y_coords, u_values):
        rounded_points = np.round(np.column_stack((x_coords, y_coords)), decimals=12)
        unique_points, inverse = np.unique(rounded_points, axis=0, return_inverse=True)
        x_unique = np.unique(unique_points[:, 0])
        y_unique = np.unique(unique_points[:, 1])
        if len(x_unique) * len(y_unique) != unique_points.shape[0]:
            return None

        summed = np.bincount(inverse, weights=np.asarray(u_values, dtype=np.float64))
        counts = np.bincount(inverse)
        averaged = summed / np.maximum(counts, 1)

        x_mesh, y_mesh = np.meshgrid(x_unique, y_unique, indexing="xy")
        grid_z = np.full((len(y_unique), len(x_unique)), np.nan, dtype=np.float64)
        x_index = {value: idx for idx, value in enumerate(x_unique)}
        y_index = {value: idx for idx, value in enumerate(y_unique)}

        for point_idx, point in enumerate(unique_points):
            grid_z[y_index[point[1]], x_index[point[0]]] = averaged[point_idx]

        return x_mesh, y_mesh, grid_z

    @staticmethod
    def _render_2d_scatter(ax, x_coords, y_coords, u_values, scatter_kwargs):
        return ax.scatter(x_coords, y_coords, c=u_values, **scatter_kwargs)

    def _finalize_2d_axes(self, ax, eq_idx: int, n_eqs: int, kwargs) -> None:
        ax.set_xlabel(self.spatial_vars[0])
        ax.set_ylabel(self.spatial_vars[1])
        title = kwargs.get("title") or ("2D Solution" if n_eqs == 1 else f"2D Solution (Variable {eq_idx})")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.15)
