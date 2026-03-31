import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.fft import dstn, idstn
from scipy.interpolate import RegularGridInterpolator


class PseudotimeVisualizeFlowMixin:
    _GHIA_RE100_Y = np.array(
        [
            1.0000,
            0.9766,
            0.9688,
            0.9609,
            0.9531,
            0.8516,
            0.7344,
            0.6172,
            0.5000,
            0.4531,
            0.2813,
            0.1719,
            0.1016,
            0.0703,
            0.0625,
            0.0547,
            0.0000,
        ]
    )
    _GHIA_RE100_U = np.array(
        [
            1.00000,
            0.84123,
            0.78871,
            0.73722,
            0.68717,
            0.23151,
            0.00332,
            -0.13641,
            -0.20581,
            -0.21090,
            -0.15662,
            -0.10150,
            -0.06434,
            -0.04775,
            -0.04192,
            -0.03717,
            0.00000,
        ]
    )
    _GHIA_RE100_X = np.array(
        [
            1.0000,
            0.9688,
            0.9609,
            0.9531,
            0.9453,
            0.9063,
            0.8594,
            0.8047,
            0.5000,
            0.2344,
            0.2266,
            0.1563,
            0.0938,
            0.0781,
            0.0703,
            0.0625,
            0.0000,
        ]
    )
    _GHIA_RE100_V = np.array(
        [
            0.00000,
            -0.05906,
            -0.07391,
            -0.08864,
            -0.10313,
            -0.16914,
            -0.22445,
            -0.24533,
            0.05454,
            0.17527,
            0.17507,
            0.16077,
            0.12317,
            0.10890,
            0.10091,
            0.09233,
            0.00000,
        ]
    )
    _BOTELLA_RE1000_STREAMLINE_LEVELS = np.array(
        [
            -1.5e-3,
            -1.0e-3,
            -5.0e-4,
            -2.5e-4,
            -1.0e-4,
            -5.0e-5,
            -1.0e-5,
            -1.0e-6,
            0.0,
            1.0e-10,
            1.0e-5,
            1.0e-4,
            1.0e-2,
            3.0e-2,
            5.0e-2,
            7.0e-2,
            9.0e-2,
            1.0e-1,
            1.1e-1,
            1.15e-1,
            1.175e-1,
        ]
    )
    _GHIA_RE1000_VORTEX_BENCHMARK = {
        "primary": {
            "x": 0.5313,
            "y": 0.5625,
            "psi": -1.17929e-1,
            "window": (0.44, 0.60, 0.48, 0.64),
            "plot_bounds": (0.0, 1.0, 0.0, 1.0),
        },
        "bottom_left_secondary": {
            "x": 0.0859,
            "y": 0.0781,
            "psi": 2.31129e-4,
            "window": (0.04, 0.14, 0.04, 0.12),
            "plot_bounds": (0.0, 0.25, 0.0, 0.25),
        },
        "bottom_right_secondary": {
            "x": 0.8594,
            "y": 0.1094,
            "psi": 1.75102e-3,
            "window": (0.80, 0.92, 0.05, 0.16),
            "plot_bounds": (0.75, 1.0, 0.0, 0.25),
        },
        "bottom_right_tertiary": {
            "x": 0.9922,
            "y": 0.0078,
            "psi": -9.31929e-8,
            "window": (0.985, 1.0, 0.0, 0.02),
            "plot_bounds": (0.95, 1.0, 0.0, 0.05),
        },
    }
    _GHIA_RE400_VORTEX_BENCHMARK = {
        "primary": {
            "x": 0.5547,
            "y": 0.6055,
            "psi": -1.13909e-1,
            "window": (0.46, 0.64, 0.52, 0.68),
            "plot_bounds": (0.0, 1.0, 0.0, 1.0),
        },
        "bottom_left_secondary": {
            "x": 0.0508,
            "y": 0.0469,
            "psi": 1.41951e-5,
            "window": (0.0, 0.10, 0.0, 0.10),
            "plot_bounds": (0.0, 0.15, 0.0, 0.15),
        },
        "bottom_right_secondary": {
            "x": 0.8906,
            "y": 0.1250,
            "psi": 6.42352e-4,
            "window": (0.82, 0.96, 0.06, 0.18),
            "plot_bounds": (0.75, 1.0, 0.0, 0.25),
        },
    }
    _RE100_ROUGH_VORTEX_BENCHMARK = {
        "primary": {
            "x": 0.6172,
            "y": 0.7344,
            "psi": 1.034000e-1,
            "window": (0.45, 0.75, 0.60, 0.82),
            "plot_bounds": (0.0, 1.0, 0.0, 1.0),
        },
        "bottom_left_secondary": {
            "x": 0.0313,
            "y": 0.0391,
            "psi": -1.750000e-6,
            "window": (0.0, 0.08, 0.0, 0.08),
            "plot_bounds": (0.0, 0.12, 0.0, 0.12),
        },
        "bottom_right_secondary": {
            "x": 0.9453,
            "y": 0.0625,
            "psi": -1.250000e-5,
            "window": (0.90, 1.0, 0.0, 0.10),
            "plot_bounds": (0.88, 1.0, 0.0, 0.12),
        },
    }

    def _save_flow_diagnostics_for_test_outputs(
        self,
        outputs,
        predictions_test,
        render_info,
        results_dir: str,
        flow_diagnostics: dict | None = None,
    ) -> None:
        if self.n_dim != 2:
            return
        direct_predictions = {}
        direct_data = None
        if flow_diagnostics:
            direct_predictions = flow_diagnostics.get("predictions", {}) or {}
            direct_data = flow_diagnostics.get("data")

        for output_name in outputs:
            if direct_data is not None and output_name in direct_predictions:
                self._save_flow_diagnostics(
                    output_name,
                    direct_data,
                    direct_predictions[output_name],
                    results_dir,
                )
                continue
            if output_name not in predictions_test:
                continue
            test_info = render_info.get(output_name, {}).get("test")
            if not test_info:
                continue
            self._save_flow_diagnostics(
                output_name,
                test_info["data"],
                predictions_test[output_name],
                results_dir,
            )

    def _save_flow_diagnostics(
        self,
        output_name: str,
        data: dict,
        prediction: np.ndarray,
        results_dir: str,
    ) -> None:
        values = self._to_numpy_cpu(prediction)
        if values is None:
            return
        values = np.asarray(values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[1] < 2:
            return

        prepared = self._prepare_plot_data(data)
        if prepared is None or "x" not in prepared:
            return

        flow_grid = self._build_flow_grid(prepared["x"], values)
        if flow_grid is None:
            print(f"  Warning: {output_name} flow diagnostics require a structured test grid; skipped.")
            return

        x_mesh, y_mesh, component_grids = flow_grid
        derived_fields = self._build_derived_flow_fields(x_mesh, y_mesh, component_grids)
        os.makedirs(results_dir, exist_ok=True)
        self._save_structured_flow_data(
            output_name,
            x_mesh,
            y_mesh,
            component_grids,
            results_dir,
            derived_fields=derived_fields,
        )
        self._write_flow_field_tecplot(
            output_name,
            x_mesh,
            y_mesh,
            component_grids,
            results_dir,
            derived_fields=derived_fields,
        )
        self._save_flow_summary_plot(output_name, x_mesh, y_mesh, component_grids, results_dir)
        self._save_centerline_profiles(output_name, x_mesh, y_mesh, component_grids, results_dir)
        self._save_streamfunction_diagnostics(
            output_name,
            x_mesh,
            y_mesh,
            component_grids,
            derived_fields,
            results_dir,
        )

    def _build_derived_flow_fields(
        self,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        component_grids: list[np.ndarray],
    ) -> dict:
        u_grid = component_grids[0]
        v_grid = component_grids[1]
        speed = np.sqrt(u_grid**2 + v_grid**2)
        vorticity = self._compute_vorticity_grid(x_mesh, y_mesh, u_grid, v_grid)
        streamfunction = self._compute_streamfunction_grid(x_mesh, y_mesh, u_grid, v_grid, vorticity)
        return {
            "speed": speed,
            "vorticity": vorticity,
            "streamfunction": streamfunction,
        }

    def _save_structured_flow_data(
        self,
        output_name: str,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        component_grids: list[np.ndarray],
        results_dir: str,
        derived_fields: dict | None = None,
    ) -> None:
        component_names = self._flow_variable_names(len(component_grids))
        save_path = os.path.join(results_dir, f"{output_name}_flow_field_structured.npz")
        payload = {
            "x_mesh": x_mesh,
            "y_mesh": y_mesh,
            "x_unique": x_mesh[0, :],
            "y_unique": y_mesh[:, 0],
            "component_names": np.asarray(component_names, dtype="<U64"),
        }
        for idx, (component_name, grid) in enumerate(zip(component_names, component_grids)):
            payload[self._sanitize_flow_component_name(component_name, idx)] = grid
        if derived_fields:
            for field_name, grid in derived_fields.items():
                if grid is None:
                    continue
                payload[self._sanitize_flow_component_name(field_name, len(payload))] = np.asarray(grid)
        np.savez(save_path, **payload)
        print(f"  Structured flow data saved: {save_path}")

    def _write_flow_field_tecplot(
        self,
        output_name: str,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        component_grids: list[np.ndarray],
        results_dir: str,
        derived_fields: dict | None = None,
    ) -> None:
        component_names = self._flow_variable_names(len(component_grids))
        ny, nx = x_mesh.shape
        save_path = os.path.join(results_dir, f"{output_name}_flow_field.dat")
        extra_fields = []
        if derived_fields:
            for field_name in ["speed", "vorticity", "streamfunction"]:
                field_values = derived_fields.get(field_name)
                if field_values is not None:
                    extra_fields.append((field_name, np.asarray(field_values)))
        with open(save_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(f'TITLE = "{output_name} Flow Field"\n')
            variable_names = ['"X"', '"Y"'] + [f'"{name}"' for name in component_names]
            variable_names.extend(f'"{field_name}"' for field_name, _ in extra_fields)
            file_obj.write(f"VARIABLES = {', '.join(variable_names)}\n")
            file_obj.write(f"ZONE I={nx}, J={ny}, F=POINT\n")
            for j in range(ny):
                for i in range(nx):
                    row_values = [x_mesh[j, i], y_mesh[j, i]]
                    row_values.extend(grid[j, i] for grid in component_grids)
                    row_values.extend(grid[j, i] for _, grid in extra_fields)
                    file_obj.write(" ".join(self._format_tecplot_value(value) for value in row_values) + "\n")
        print(f"  Tecplot flow data saved: {save_path}")

    def _flow_variable_names(self, count: int) -> list[str]:
        configured_names = list(getattr(self.config, "vars_list", []) or [])
        names = [str(name) for name in configured_names[:count]]
        while len(names) < count:
            names.append(f"var_{len(names)}")
        return names

    @staticmethod
    def _sanitize_flow_component_name(component_name: str, index: int) -> str:
        sanitized = "".join(char if (char.isalnum() or char == "_") else "_" for char in str(component_name).strip())
        if not sanitized:
            sanitized = f"field_{index}"
        if sanitized[0].isdigit():
            sanitized = f"field_{index}_{sanitized}"
        return sanitized

    @staticmethod
    def _format_tecplot_value(value: float) -> str:
        if np.isnan(value):
            return "nan"
        return f"{float(value):.12e}"

    def _build_flow_grid(self, coords: np.ndarray, values: np.ndarray):
        plotter = getattr(self, "dimensional_visualizer", None)
        if plotter is None:
            return None

        first_payload = plotter._build_structured_grid_payload(coords[:, 0], coords[:, 1], values[:, 0])
        if first_payload is None:
            return None

        x_mesh, y_mesh, first_grid = first_payload
        component_grids = [first_grid]
        for eq_idx in range(1, values.shape[1]):
            payload = plotter._build_structured_grid_payload(coords[:, 0], coords[:, 1], values[:, eq_idx])
            if payload is None:
                return None
            component_grids.append(payload[2])
        return x_mesh, y_mesh, component_grids

    @staticmethod
    def _is_uniform_axis(axis_values: np.ndarray) -> bool:
        if axis_values.size < 3:
            return True
        deltas = np.diff(np.asarray(axis_values, dtype=np.float64))
        return np.allclose(deltas, deltas[0], rtol=1e-8, atol=1e-12)

    def _compute_vorticity_grid(
        self,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
    ) -> np.ndarray:
        x_unique = x_mesh[0, :]
        y_unique = y_mesh[:, 0]
        dv_dx = np.gradient(v_grid, x_unique, axis=1, edge_order=2)
        du_dy = np.gradient(u_grid, y_unique, axis=0, edge_order=2)
        return dv_dx - du_dy

    def _compute_streamfunction_grid(
        self,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
        vorticity: np.ndarray | None = None,
    ) -> np.ndarray | None:
        x_unique = x_mesh[0, :]
        y_unique = y_mesh[:, 0]
        if vorticity is None:
            vorticity = self._compute_vorticity_grid(x_mesh, y_mesh, u_grid, v_grid)

        if self._is_uniform_axis(x_unique) and self._is_uniform_axis(y_unique):
            try:
                return self._solve_streamfunction_poisson_uniform(x_unique, y_unique, vorticity)
            except Exception as exc:
                print(f"  Warning: uniform streamfunction solve failed, fallback to path integration: {exc}")
        return self._integrate_streamfunction_from_velocity(x_unique, y_unique, u_grid, v_grid)

    @staticmethod
    def _solve_streamfunction_poisson_uniform(
        x_unique: np.ndarray,
        y_unique: np.ndarray,
        vorticity: np.ndarray,
    ) -> np.ndarray:
        nx = x_unique.size
        ny = y_unique.size
        psi = np.zeros((ny, nx), dtype=np.float64)
        if nx <= 2 or ny <= 2:
            return psi

        dx = float(x_unique[1] - x_unique[0])
        dy = float(y_unique[1] - y_unique[0])
        rhs = -np.asarray(vorticity[1:-1, 1:-1], dtype=np.float64)
        rhs_hat = dstn(rhs, type=1, norm="ortho")

        kx = np.arange(1, nx - 1, dtype=np.float64)
        ky = np.arange(1, ny - 1, dtype=np.float64)
        lambda_x = 2.0 * (np.cos(np.pi * kx / (nx - 1)) - 1.0) / (dx * dx)
        lambda_y = 2.0 * (np.cos(np.pi * ky / (ny - 1)) - 1.0) / (dy * dy)
        denom = lambda_y[:, None] + lambda_x[None, :]
        psi_hat = rhs_hat / denom
        psi[1:-1, 1:-1] = idstn(psi_hat, type=1, norm="ortho")
        return psi

    @staticmethod
    def _integrate_streamfunction_from_velocity(
        x_unique: np.ndarray,
        y_unique: np.ndarray,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
    ) -> np.ndarray:
        psi_y = np.zeros_like(u_grid, dtype=np.float64)
        for row_idx in range(1, len(y_unique)):
            dy = float(y_unique[row_idx] - y_unique[row_idx - 1])
            psi_y[row_idx, :] = psi_y[row_idx - 1, :] + 0.5 * dy * (
                u_grid[row_idx, :] + u_grid[row_idx - 1, :]
            )

        psi_x = np.zeros_like(v_grid, dtype=np.float64)
        for col_idx in range(1, len(x_unique)):
            dx = float(x_unique[col_idx] - x_unique[col_idx - 1])
            psi_x[:, col_idx] = psi_x[:, col_idx - 1] - 0.5 * dx * (
                v_grid[:, col_idx] + v_grid[:, col_idx - 1]
            )

        psi = 0.5 * (psi_y + psi_x)
        boundary_values = np.concatenate([psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]])
        psi -= float(np.nanmean(boundary_values))
        return psi

    def _save_streamfunction_diagnostics(
        self,
        output_name: str,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        component_grids: list[np.ndarray],
        derived_fields: dict,
        results_dir: str,
    ) -> None:
        streamfunction = derived_fields.get("streamfunction")
        if streamfunction is None:
            return

        x_ref, y_ref, psi_ref = self._refine_scalar_grid(x_mesh, y_mesh, streamfunction)
        levels = self._streamfunction_levels(psi_ref)
        benchmark = self._streamfunction_benchmark_data()
        detections = self._detect_streamfunction_vortices(x_ref, y_ref, psi_ref)

        self._save_streamfunction_global_plot(
            output_name,
            x_ref,
            y_ref,
            psi_ref,
            levels,
            detections,
            benchmark,
            results_dir,
        )
        self._save_corner_vortex_plot(
            output_name,
            x_ref,
            y_ref,
            psi_ref,
            detections,
            benchmark,
            results_dir,
        )
        self._save_streamfunction_benchmark_report(
            output_name,
            levels,
            detections,
            benchmark,
            results_dir,
        )
        self._save_vortex_comparison_plot(
            output_name,
            x_ref,
            y_ref,
            psi_ref,
            detections,
            benchmark,
            results_dir,
        )

    def _refine_scalar_grid(
        self,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        scalar_grid: np.ndarray,
    ):
        x_unique = x_mesh[0, :]
        y_unique = y_mesh[:, 0]
        target_nx = self._refined_axis_size(x_unique.size)
        target_ny = self._refined_axis_size(y_unique.size)
        if target_nx == x_unique.size and target_ny == y_unique.size:
            return x_mesh, y_mesh, np.asarray(scalar_grid)

        x_ref = np.linspace(float(x_unique.min()), float(x_unique.max()), target_nx)
        y_ref = np.linspace(float(y_unique.min()), float(y_unique.max()), target_ny)
        x_ref_mesh, y_ref_mesh = np.meshgrid(x_ref, y_ref, indexing="xy")
        interpolator = RegularGridInterpolator(
            (y_unique, x_unique),
            np.asarray(scalar_grid),
            bounds_error=False,
            fill_value=np.nan,
        )
        values = interpolator(np.column_stack([y_ref_mesh.ravel(), x_ref_mesh.ravel()])).reshape(target_ny, target_nx)
        return x_ref_mesh, y_ref_mesh, values

    @staticmethod
    def _refined_axis_size(size: int) -> int:
        if size <= 1:
            return size
        return min(max(size, 2 * size - 1), 801)

    def _streamfunction_levels(self, psi_grid: np.ndarray) -> np.ndarray:
        re_value = self._reynolds_number()
        if re_value is not None and abs(re_value - 1000.0) < 1e-12:
            return self._BOTELLA_RE1000_STREAMLINE_LEVELS
        return self._build_log_contour_levels(psi_grid, positive_count=12, negative_count=8)

    @staticmethod
    def _build_log_contour_levels(
        values: np.ndarray,
        positive_count: int = 10,
        negative_count: int = 10,
    ) -> np.ndarray:
        value_min = float(np.nanmin(values))
        value_max = float(np.nanmax(values))
        levels = []
        if value_min < 0.0:
            neg_abs = np.geomspace(max(abs(value_min) * 1e-6, 1e-12), max(abs(value_min) * 0.98, 1e-12), negative_count)
            levels.extend((-neg_abs[::-1]).tolist())
        levels.append(0.0)
        if value_max > 0.0:
            pos = np.geomspace(max(value_max * 1e-6, 1e-12), max(value_max * 0.98, 1e-12), positive_count)
            levels.extend(pos.tolist())
        levels_array = np.asarray(sorted(set(levels)), dtype=np.float64)
        return levels_array[np.isfinite(levels_array)]

    def _save_streamfunction_global_plot(
        self,
        output_name: str,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        psi_grid: np.ndarray,
        levels: np.ndarray,
        detections: dict,
        benchmark: dict | None,
        results_dir: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 8.0))
        contour = ax.contour(
            x_mesh,
            y_mesh,
            psi_grid,
            levels=levels,
            colors="k",
            linewidths=0.9,
        )
        if len(contour.levels) <= 12:
            ax.clabel(contour, inline=True, fontsize=8, fmt="%.2e")
        self._annotate_streamfunction_vortices(ax, detections, benchmark)
        ax.set_xlabel(self.config.spatial_vars[0])
        ax.set_ylabel(self.config.spatial_vars[1])
        ax.set_title(f"{output_name.capitalize()} streamfunction contours")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.15)
        save_path = os.path.join(results_dir, f"{output_name}_streamfunction_global.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=getattr(self.config.visualization, "figure_dpi", 120), bbox_inches="tight")
        plt.close(fig)
        print(f"  Streamfunction contours saved: {save_path}")

    def _save_corner_vortex_plot(
        self,
        output_name: str,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        psi_grid: np.ndarray,
        detections: dict,
        benchmark: dict | None,
        results_dir: str,
    ) -> None:
        windows = self._corner_plot_windows(benchmark)
        rows = 1 if len(windows) <= 2 else 2
        cols = 2 if len(windows) > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        axes = np.atleast_1d(axes).ravel()
        figure_dpi = getattr(self.config.visualization, "figure_dpi", 120)

        for ax, (vortex_name, title, bounds) in zip(axes, windows):
            x_min, x_max, y_min, y_max = bounds
            mask = (
                (x_mesh >= x_min)
                & (x_mesh <= x_max)
                & (y_mesh >= y_min)
                & (y_mesh <= y_max)
            )
            if np.count_nonzero(mask) < 16:
                ax.text(0.5, 0.5, "Insufficient points", ha="center", va="center", transform=ax.transAxes)
                continue

            psi_local = np.where(mask, psi_grid, np.nan)
            finite_local = psi_local[np.isfinite(psi_local)]
            if finite_local.size == 0 or np.nanmax(finite_local) - np.nanmin(finite_local) < 1e-12:
                ax.text(0.5, 0.5, "No resolvable vortex", ha="center", va="center", transform=ax.transAxes)
            else:
                levels = self._build_log_contour_levels(psi_local, positive_count=6, negative_count=6)
                if levels.size >= 2:
                    ax.contour(x_mesh, y_mesh, psi_local, levels=levels, colors="k", linewidths=0.9)
                ax.contourf(x_mesh, y_mesh, psi_local, levels=30, cmap="Greys", alpha=0.12)

            detected = detections.get(vortex_name)
            if detected is not None:
                ax.plot(detected["x"], detected["y"], "ro", ms=5, label="DeePoly")
            ref_item = None if benchmark is None else benchmark.get(vortex_name)
            if ref_item is not None:
                ax.plot(ref_item["x"], ref_item["y"], "kx", ms=6, mew=1.3, label="Benchmark")
            if detected is not None or ref_item is not None:
                ax.legend(loc="best", fontsize=8)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(title)
            ax.set_xlabel(self.config.spatial_vars[0])
            ax.set_ylabel(self.config.spatial_vars[1])
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.15)

        for ax in axes[len(windows):]:
            ax.axis("off")

        fig.tight_layout()
        save_path = os.path.join(results_dir, f"{output_name}_corner_vortices.png")
        fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Corner-vortex diagnostics saved: {save_path}")

    @staticmethod
    def _annotate_streamfunction_vortices(ax, detections: dict, benchmark: dict | None) -> None:
        primary = detections.get("primary")
        if primary is not None:
            ax.plot(primary["x"], primary["y"], "ro", ms=5)
        if benchmark is not None and benchmark.get("primary") is not None:
            ref_primary = benchmark["primary"]
            ax.plot(ref_primary["x"], ref_primary["y"], "kx", ms=6, mew=1.3)

    def _corner_plot_windows(self, benchmark: dict | None):
        if benchmark is None:
            return [
                ("bottom_left_secondary", "Bottom-left Corner", (0.0, 0.25, 0.0, 0.25)),
                ("bottom_right_secondary", "Bottom-right Corner", (0.75, 1.0, 0.0, 0.25)),
            ]
        titles = {
            "bottom_left_secondary": "Bottom-left Corner",
            "bottom_right_secondary": "Bottom-right Corner",
            "bottom_left_tertiary": "Bottom-left Near-corner",
            "bottom_right_tertiary": "Bottom-right Near-corner",
        }
        windows = []
        for vortex_name in [
            "bottom_left_secondary",
            "bottom_right_secondary",
            "bottom_left_tertiary",
            "bottom_right_tertiary",
        ]:
            item = benchmark.get(vortex_name)
            if item is None:
                continue
            windows.append((vortex_name, titles[vortex_name], tuple(item["plot_bounds"])))
        return windows

    def _save_vortex_comparison_plot(
        self,
        output_name: str,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        psi_grid: np.ndarray,
        detections: dict,
        benchmark: dict | None,
        results_dir: str,
    ) -> None:
        if benchmark is None:
            return

        figure_dpi = getattr(self.config.visualization, "figure_dpi", 120)
        fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.5))
        legend_handles = [
            Line2D([], [], color="red", marker="o", linestyle="None", markersize=5, label="DeePoly"),
            Line2D([], [], color="black", marker="x", linestyle="None", markersize=6, label="Reference"),
        ]

        global_levels = self._streamfunction_levels(psi_grid)
        axes[0].contour(x_mesh, y_mesh, psi_grid, levels=global_levels, colors="k", linewidths=0.8)
        self._annotate_streamfunction_vortices(axes[0], detections, benchmark)
        self._annotate_vortex_metrics(
            axes[0],
            "primary",
            detections.get("primary"),
            benchmark.get("primary"),
            anchor=(0.02, 0.98),
            ha="left",
            va="top",
            fontsize=7.8,
        )
        axes[0].legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=8,
            framealpha=0.92,
            borderaxespad=0.0,
        )
        axes[0].set_xlim(0.0, 1.0)
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("Primary Vortex")
        axes[0].set_xlabel(self.config.spatial_vars[0])
        axes[0].set_ylabel(self.config.spatial_vars[1])
        axes[0].set_aspect("equal")
        axes[0].grid(True, linestyle="--", alpha=0.15)

        for ax, vortex_name in zip(axes[1:], ["bottom_left_secondary", "bottom_right_secondary"]):
            ref_item = benchmark.get(vortex_name)
            if ref_item is None:
                ax.axis("off")
                continue
            x_min, x_max, y_min, y_max = ref_item["plot_bounds"]
            mask = (
                (x_mesh >= x_min)
                & (x_mesh <= x_max)
                & (y_mesh >= y_min)
                & (y_mesh <= y_max)
            )
            psi_local = np.where(mask, psi_grid, np.nan)
            levels = self._build_log_contour_levels(psi_local, positive_count=6, negative_count=6)
            if levels.size >= 2:
                ax.contour(x_mesh, y_mesh, psi_local, levels=levels, colors="k", linewidths=0.9)
            current_item = detections.get(vortex_name)
            if current_item is not None:
                ax.plot(current_item["x"], current_item["y"], "ro", ms=5, label="DeePoly")
            ax.plot(ref_item["x"], ref_item["y"], "kx", ms=6, mew=1.3, label="Reference")
            self._annotate_vortex_metrics(
                ax,
                vortex_name,
                current_item,
                ref_item,
                anchor=(0.02, 0.98),
                ha="left",
                fontsize=7.8,
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title("BL Corner" if "left" in vortex_name else "BR Corner")
            ax.set_xlabel(self.config.spatial_vars[0])
            ax.set_ylabel(self.config.spatial_vars[1])
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.15)
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),
                fontsize=8,
                framealpha=0.92,
                borderaxespad=0.0,
            )

        fig.subplots_adjust(left=0.05, right=0.985, bottom=0.12, top=0.9, wspace=0.34)
        save_path = os.path.join(results_dir, f"{output_name}_vortex_comparison.png")
        pdf_path = os.path.join(results_dir, f"{output_name}_vortex_comparison.pdf")
        fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Vortex comparison saved: {save_path}")
        print(f"  Vortex comparison saved: {pdf_path}")

    @staticmethod
    def _annotate_vortex_metrics(
        ax,
        vortex_name: str,
        current_item: dict | None,
        ref_item: dict | None,
        anchor=(0.02, 0.98),
        ha: str = "left",
        va: str = "top",
        fontsize: float = 8.5,
    ) -> None:
        lines = []
        if current_item is not None:
            lines.append(
                "D: "
                f"({current_item['x']:.4f}, {current_item['y']:.4f})"
            )
            lines.append(f"   |psi|={abs(float(current_item['psi'])):.3e}")
        if ref_item is not None:
            lines.append(
                "R: "
                f"({ref_item['x']:.4f}, {ref_item['y']:.4f})"
            )
            lines.append(f"   |psi|={abs(float(ref_item['psi'])):.3e}")
        if current_item is not None and ref_item is not None:
            lines.append(f"dx={abs(float(current_item['x']) - float(ref_item['x'])):.3e}")
            lines.append(f"dy={abs(float(current_item['y']) - float(ref_item['y'])):.3e}")
        ax.text(
            anchor[0],
            anchor[1],
            "\n".join(lines),
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=fontsize,
            family="monospace",
            multialignment="left",
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.6",
                "alpha": 0.92,
            },
        )

    def _save_streamfunction_benchmark_report(
        self,
        output_name: str,
        levels: np.ndarray,
        detections: dict,
        benchmark: dict | None,
        results_dir: str,
    ) -> None:
        comparisons = {}
        if benchmark is not None:
            for vortex_name, ref_item in benchmark.items():
                current_item = detections.get(vortex_name)
                if current_item is None:
                    continue
                comparisons[vortex_name] = {
                    "x_abs_error": abs(float(current_item["x"]) - float(ref_item["x"])),
                    "y_abs_error": abs(float(current_item["y"]) - float(ref_item["y"])),
                    "psi_abs_error": abs(abs(float(current_item["psi"])) - abs(float(ref_item["psi"]))),
                }

        payload = {
            "output": output_name,
            "reynolds_number": self._reynolds_number(),
            "benchmark_source": self._streamfunction_benchmark_source(),
            "streamfunction_levels": levels.tolist(),
            "detected_vortices": detections,
            "benchmark_vortices": benchmark,
            "comparisons": comparisons,
        }
        save_path = os.path.join(results_dir, f"{output_name}_vortex_benchmark.json")
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"  Vortex benchmark saved: {save_path}")

    def _detect_streamfunction_vortices(
        self,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        psi_grid: np.ndarray,
    ) -> dict:
        benchmark = self._streamfunction_benchmark_data()
        if benchmark is None:
            return {}
        detections = {}
        global_scale = max(float(np.nanmax(np.abs(psi_grid))), 1e-12)

        primary_item = benchmark.get("primary")
        if primary_item is None:
            return {}

        primary_detection = self._detect_single_vortex(
            x_mesh,
            y_mesh,
            psi_grid,
            primary_item["window"],
            mode="abs",
            global_scale=global_scale,
        )
        if primary_detection is not None:
            detections["primary"] = primary_detection

        sign_flip = 1.0
        if primary_detection is not None and primary_item["psi"] != 0.0:
            sign_flip = np.sign(primary_detection["psi"]) / np.sign(primary_item["psi"])
            if not np.isfinite(sign_flip) or sign_flip == 0.0:
                sign_flip = 1.0

        for vortex_name, ref_item in benchmark.items():
            if vortex_name == "primary":
                continue
            expected_sign = np.sign(ref_item["psi"]) * sign_flip
            mode = "max" if expected_sign >= 0.0 else "min"
            detected = self._detect_single_vortex(
                x_mesh,
                y_mesh,
                psi_grid,
                ref_item["window"],
                mode=mode,
                global_scale=global_scale,
            )
            if detected is not None:
                detections[vortex_name] = detected
        return detections

    @staticmethod
    def _detect_single_vortex(
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        psi_grid: np.ndarray,
        bounds,
        mode: str,
        global_scale: float,
    ) -> dict | None:
        x_min, x_max, y_min, y_max = bounds
        mask = (
            (x_mesh >= x_min)
            & (x_mesh <= x_max)
            & (y_mesh >= y_min)
            & (y_mesh <= y_max)
        )
        if np.count_nonzero(mask) == 0:
            return None
        psi_values = psi_grid[mask]
        if psi_values.size == 0:
            return None

        if mode == "abs":
            flat_index = int(np.nanargmax(np.abs(psi_values)))
            local_value = float(psi_values[flat_index])
        elif mode == "max":
            flat_index = int(np.nanargmax(psi_values))
            local_value = float(psi_values[flat_index])
        else:
            flat_index = int(np.nanargmin(psi_values))
            local_value = float(psi_values[flat_index])

        if abs(local_value) < global_scale * 1e-8:
            return None
        local_indices = np.argwhere(mask)
        row_idx, col_idx = local_indices[flat_index]
        return {
            "x": float(x_mesh[row_idx, col_idx]),
            "y": float(y_mesh[row_idx, col_idx]),
            "psi": local_value,
        }

    def _streamfunction_benchmark_data(self) -> dict | None:
        re_value = self._reynolds_number()
        if re_value is None:
            return None
        if abs(re_value - 100.0) < 1e-12:
            return self._RE100_ROUGH_VORTEX_BENCHMARK
        if abs(re_value - 400.0) < 1e-12:
            return self._GHIA_RE400_VORTEX_BENCHMARK
        if abs(re_value - 1000.0) < 1e-12:
            return self._GHIA_RE1000_VORTEX_BENCHMARK
        return None

    def _streamfunction_benchmark_source(self) -> str | None:
        re_value = self._reynolds_number()
        if re_value is None:
            return None
        if abs(re_value - 100.0) < 1e-12:
            return "Monaco thesis Table 3.4 (compiled Re=100 cavity vortex data)"
        if abs(re_value - 400.0) < 1e-12:
            return "Ghia, Ghia, and Shin (1982) Table V"
        if abs(re_value - 1000.0) < 1e-12:
            return "Ghia, Ghia, and Shin (1982) Table V"
        return None

    def _reynolds_number(self) -> float | None:
        constants = getattr(self.config, "constants", {})
        if not isinstance(constants, dict):
            return None
        re_value = constants.get("Re")
        if re_value is None:
            return None
        try:
            return float(re_value)
        except (TypeError, ValueError):
            return None

    def _save_flow_summary_plot(
        self,
        output_name: str,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        component_grids: list[np.ndarray],
        results_dir: str,
    ) -> None:
        u_grid = component_grids[0]
        v_grid = component_grids[1]
        p_grid = component_grids[2] if len(component_grids) > 2 else None
        p_grid_centered = None if p_grid is None else (p_grid - np.nanmean(p_grid))
        speed = np.sqrt(u_grid**2 + v_grid**2)
        x_unique = x_mesh[0, :]
        y_unique = y_mesh[:, 0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        title_prefix = output_name.capitalize()
        cmap = getattr(self.config.visualization, "colormap", "viridis")

        c0 = axes[0, 0].contourf(x_mesh, y_mesh, u_grid, levels=40, cmap=cmap)
        fig.colorbar(c0, ax=axes[0, 0], label="u")
        axes[0, 0].set_title(f"{title_prefix} u")

        c1 = axes[0, 1].contourf(x_mesh, y_mesh, v_grid, levels=40, cmap=cmap)
        fig.colorbar(c1, ax=axes[0, 1], label="v")
        axes[0, 1].set_title(f"{title_prefix} v")

        c2 = axes[1, 0].contourf(x_mesh, y_mesh, speed, levels=40, cmap=cmap)
        fig.colorbar(c2, ax=axes[1, 0], label="|V|")
        axes[1, 0].streamplot(
            x_unique,
            y_unique,
            u_grid,
            v_grid,
            color="k",
            linewidth=0.8,
            density=2.2,
            arrowsize=0.8,
        )
        axes[1, 0].set_title(f"{title_prefix} speed + streamlines")

        if p_grid_centered is not None:
            c3 = axes[1, 1].contourf(x_mesh, y_mesh, p_grid_centered, levels=40, cmap=cmap)
            fig.colorbar(c3, ax=axes[1, 1], label="p - <p>")
            axes[1, 1].set_title(f"{title_prefix} pressure (mean-shifted)")
        else:
            axes[1, 1].quiver(
                x_mesh[::8, ::8],
                y_mesh[::8, ::8],
                u_grid[::8, ::8],
                v_grid[::8, ::8],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.002,
            )
            axes[1, 1].set_title(f"{title_prefix} velocity vectors")

        for ax in axes.ravel():
            ax.set_xlabel(self.config.spatial_vars[0])
            ax.set_ylabel(self.config.spatial_vars[1])
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.15)

        fig.tight_layout()
        save_path = os.path.join(results_dir, f"{output_name}_flow_summary.png")
        fig.savefig(save_path, dpi=getattr(self.config.visualization, "figure_dpi", 120), bbox_inches="tight")
        plt.close(fig)
        print(f"  Flow summary saved: {save_path}")

    def _save_centerline_profiles(
        self,
        output_name: str,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        component_grids: list[np.ndarray],
        results_dir: str,
    ) -> None:
        u_grid = component_grids[0]
        v_grid = component_grids[1]
        x_unique = x_mesh[0, :]
        y_unique = y_mesh[:, 0]
        u_interp = RegularGridInterpolator((y_unique, x_unique), u_grid, bounds_error=False, fill_value=None)
        v_interp = RegularGridInterpolator((y_unique, x_unique), v_grid, bounds_error=False, fill_value=None)
        y_line = np.linspace(y_unique.min(), y_unique.max(), max(len(y_unique), 401))
        x_line = np.linspace(x_unique.min(), x_unique.max(), max(len(x_unique), 401))
        u_centerline = u_interp(np.column_stack([y_line, np.full_like(y_line, 0.5)]))
        v_centerline = v_interp(np.column_stack([np.full_like(x_line, 0.5), x_line]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(u_centerline, y_line, linewidth=2.0, label="DeePoly")
        axes[0].set_xlabel("u")
        axes[0].set_ylabel("y")
        axes[0].set_title("Vertical Centerline x=0.5")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(x_line, v_centerline, linewidth=2.0, label="DeePoly")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("v")
        axes[1].set_title("Horizontal Centerline y=0.5")
        axes[1].grid(True, alpha=0.25)

        if self._is_re100_lid_cavity():
            axes[0].plot(self._GHIA_RE100_U, self._GHIA_RE100_Y, "ko", ms=3.5, label="Ghia 1982")
            axes[1].plot(self._GHIA_RE100_X, self._GHIA_RE100_V, "ko", ms=3.5, label="Ghia 1982")

        axes[0].legend(loc="best")
        axes[1].legend(loc="best")
        fig.tight_layout()
        save_path = os.path.join(results_dir, f"{output_name}_centerline_profiles.png")
        fig.savefig(save_path, dpi=getattr(self.config.visualization, "figure_dpi", 120), bbox_inches="tight")
        plt.close(fig)
        print(f"  Centerline profiles saved: {save_path}")

    def _is_re100_lid_cavity(self) -> bool:
        re_value = self._reynolds_number()
        return re_value is not None and abs(re_value - 100.0) < 1e-12
