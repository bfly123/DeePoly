import os
from typing import Dict, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class PseudoTimeAnimationMixin:
    def update_animation_data(self, iteration, residual, U, data, skip, reference_solution=None):
        if skip:
            self._anim_skip = max(1, int(skip))
        self._record_animation_step(iteration, residual, U, data, reference_solution)

    def _init_animation_state(self):
        vis_cfg = getattr(self.config, "visualization", None)
        self._anim_enabled = bool(vis_cfg and (vis_cfg.save_animation or vis_cfg.realtime_visualization))
        self._anim_skip = getattr(vis_cfg, "animation_skip", 1) if vis_cfg else 1
        self._anim_iterations = []
        self._anim_residuals = []
        self._anim_snapshots = []
        self._anim_coords = None
        self._anim_reference_values = None
        self._anim_colormap = getattr(vis_cfg, "colormap", "RdBu_r") if vis_cfg else "RdBu_r"
        self._realtime_mode_active = False
        self._realtime_fig = None
        self._realtime_axes = {}
        self._realtime_handles = {}
        self._anim_dataset_kind = "train"

    def reset_animation(self):
        self._init_animation_state()

    def _record_animation_step(
        self,
        iteration: int,
        residual: float,
        U_pred,
        data_train: Dict,
        reference_solution: Optional[Dict],
    ):
        vis_cfg = getattr(self.config, "visualization", None)
        if not self._should_capture_animation_step(iteration, vis_cfg):
            return

        coords = self._prepare_animation_coords(data_train)
        if coords is None:
            return

        values = self._normalize_animation_values(U_pred)
        if values is None:
            return

        self._sync_animation_coords(coords, reference_solution)
        self._anim_iterations.append(int(iteration))
        self._anim_residuals.append(float(residual))
        self._anim_snapshots.append(values.astype(np.float32, copy=True))

        if vis_cfg.realtime_visualization:
            self._update_realtime_display(values)

    def _should_capture_animation_step(self, iteration: int, vis_cfg) -> bool:
        if not vis_cfg or not (vis_cfg.save_animation or vis_cfg.realtime_visualization):
            return False
        return iteration % max(1, self._anim_skip) == 0

    def _prepare_animation_coords(self, data_train: Dict):
        prepared = self._prepare_plot_data(data_train)
        if prepared is None or "x" not in prepared:
            return None
        vis_cfg = getattr(self.config, "visualization", None)
        if vis_cfg and getattr(vis_cfg, "should_plot_test", None) and vis_cfg.should_plot_test():
            self._anim_dataset_kind = "test"
        else:
            self._anim_dataset_kind = "train"
        return prepared["x"]

    def _normalize_animation_values(self, U_pred):
        values = self._to_numpy_cpu(U_pred)
        if values is None:
            return None
        values = np.asarray(values)
        return values.reshape(-1, 1) if values.ndim == 1 else values

    def _sync_animation_coords(self, coords: np.ndarray, reference_solution: Optional[Dict]) -> None:
        if self._anim_coords is None:
            self._anim_coords = coords.copy()
            self._anim_reference_values = self._evaluate_reference_at_coords(coords, reference_solution)
            return
        if coords.shape != self._anim_coords.shape or not np.allclose(coords, self._anim_coords):
            self._anim_coords = coords.copy()
            self._anim_reference_values = self._evaluate_reference_at_coords(coords, reference_solution)

    def _evaluate_reference_at_coords(self, coords: np.ndarray, reference_solution: Optional[Dict]):
        if not reference_solution:
            return None
        try:
            if self.n_dim == 1:
                return self._evaluate_1d_reference(coords, reference_solution)
            if self.n_dim == 2:
                return self._evaluate_2d_reference(coords, reference_solution)
        except Exception as exc:
            print(f"  Warning: Failed to evaluate reference on training coords: {exc}")
        return None

    def _evaluate_1d_reference(self, coords: np.ndarray, reference_solution: Dict):
        x_ref = reference_solution.get("x_ref")
        u_ref = reference_solution.get("u_ref")
        if x_ref is None or u_ref is None:
            return None
        coord_flat = coords[:, 0] if coords.ndim > 1 else coords
        ref_vals = np.interp(coord_flat, np.asarray(x_ref), np.asarray(u_ref))
        return ref_vals.reshape(-1, 1)

    def _evaluate_2d_reference(self, coords: np.ndarray, reference_solution: Dict):
        x_ref = reference_solution.get("x_ref")
        y_ref = reference_solution.get("y_ref")
        u_ref = reference_solution.get("u_ref")
        if x_ref is None or y_ref is None or u_ref is None:
            return None

        x_ref = np.asarray(x_ref)
        y_ref = np.asarray(y_ref)
        u_grid = self._reshape_reference_grid(x_ref, y_ref, np.asarray(u_ref))
        if u_grid is None:
            return None

        interpolator = RegularGridInterpolator((x_ref, y_ref), u_grid, bounds_error=False, fill_value=None)
        ref_vals = interpolator(coords[:, :2])
        return ref_vals.reshape(-1, 1)

    def _update_realtime_display(self, values: np.ndarray):
        coords = self._anim_coords
        if coords is None:
            return

        if not self._realtime_mode_active:
            self._init_realtime_plot()

        value_to_plot = values[:, 0] if values.ndim > 1 else values.squeeze()
        if self.n_dim == 1:
            self._update_realtime_display_1d(coords, value_to_plot)
        else:
            self._update_realtime_display_2d(coords, value_to_plot)

        residual_line = self._realtime_handles["residual_line"]
        residual_line.set_data(self._anim_iterations, self._anim_residuals)
        ax_res = self._realtime_axes["residual"]
        ax_res.relim()
        ax_res.autoscale_view()
        ax_res.set_xlim(0, max(1, max(self._anim_iterations)) if self._anim_iterations else 1)

        self._realtime_fig.canvas.draw()
        self._realtime_fig.canvas.flush_events()

    def _update_realtime_display_1d(self, coords: np.ndarray, value_to_plot: np.ndarray) -> None:
        x_flat = coords[:, 0] if coords.ndim > 1 else coords
        order = np.argsort(x_flat)
        x_sorted = x_flat[order]
        pred_sorted = value_to_plot[order]
        self._realtime_handles["pred_line"].set_data(x_sorted, pred_sorted)
        if self._anim_reference_values is not None:
            ref_sorted = self._anim_reference_values.squeeze()[order]
            self._realtime_handles["ref_line"].set_data(x_sorted, ref_sorted)

        ax_sol = self._realtime_axes["solution"]
        ax_sol.relim()
        ax_sol.autoscale_view()

    def _update_realtime_display_2d(self, coords: np.ndarray, value_to_plot: np.ndarray) -> None:
        scatter_pred = self._realtime_handles["pred_scatter"]
        scatter_pred.set_offsets(coords[:, :2])
        scatter_pred.set_array(value_to_plot.astype(np.float32))
        self._set_realtime_axis_limits(self._realtime_axes["prediction"], coords)
        if self._anim_reference_values is not None:
            scatter_ref = self._realtime_handles["ref_scatter"]
            scatter_ref.set_offsets(coords[:, :2])
            scatter_ref.set_array(self._anim_reference_values.squeeze().astype(np.float32))
            if self._realtime_axes["reference"] is not None:
                self._set_realtime_axis_limits(self._realtime_axes["reference"], coords)

    def _init_realtime_plot(self):
        import matplotlib.pyplot as plt

        if self._realtime_mode_active:
            return

        has_reference = self._anim_reference_values is not None
        if self.n_dim == 1:
            self._init_realtime_plot_1d(has_reference)
        else:
            self._init_realtime_plot_2d(has_reference)

        self._realtime_mode_active = True
        self._realtime_fig.tight_layout(pad=2.0)
        plt.ion()
        try:
            self._realtime_fig.canvas.manager.show()
        except Exception:
            try:
                self._realtime_fig.show()
            except Exception:
                pass

    def _init_realtime_plot_1d(self, has_reference: bool) -> None:
        import matplotlib.pyplot as plt

        self._realtime_fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_sol, ax_res = axes
        vars_list = getattr(self.config, "vars_list", ["u"])
        ax_sol.set_xlabel(self.config.spatial_vars[0])
        ax_sol.set_ylabel(vars_list[0])
        ax_sol.grid(True, alpha=0.3)

        pred_line, = ax_sol.plot([], [], "b-", linewidth=2, label="Prediction")
        ref_line = None
        if has_reference:
            ref_line, = ax_sol.plot([], [], "r--", linewidth=2, label="Reference")
            ax_sol.legend(loc="best")

        ax_res.set_xlabel("Iteration")
        ax_res.set_ylabel("Residual")
        ax_res.set_yscale("log")
        ax_res.grid(True, alpha=0.3)
        residual_line, = ax_res.plot([], [], "r-", linewidth=2)

        self._realtime_axes = {"solution": ax_sol, "residual": ax_res}
        self._realtime_handles = {"pred_line": pred_line, "ref_line": ref_line, "residual_line": residual_line}

    def _init_realtime_plot_2d(self, has_reference: bool) -> None:
        import matplotlib.pyplot as plt

        num_cols = 3 if has_reference else 2
        self._realtime_fig, axes = plt.subplots(1, num_cols, figsize=(18, 6))
        ax_idx = 0
        ax_pred = axes[ax_idx]
        scat_pred = ax_pred.scatter([], [], c=[], cmap=self._anim_colormap, s=20, edgecolors="none")
        self._realtime_fig.colorbar(scat_pred, ax=ax_pred, label="Prediction")
        ax_pred.set_xlabel(self.config.spatial_vars[0])
        ax_pred.set_ylabel(self.config.spatial_vars[1])
        ax_pred.set_title("Prediction")
        ax_pred.grid(True, linestyle="--", alpha=0.2)
        ax_idx += 1

        ax_ref = None
        scat_ref = None
        if has_reference:
            ax_ref = axes[ax_idx]
            scat_ref = ax_ref.scatter([], [], c=[], cmap=self._anim_colormap, s=20, edgecolors="none")
            self._realtime_fig.colorbar(scat_ref, ax=ax_ref, label="Reference")
            ax_ref.set_xlabel(self.config.spatial_vars[0])
            ax_ref.set_ylabel(self.config.spatial_vars[1])
            ax_ref.set_title("Reference")
            ax_ref.grid(True, linestyle="--", alpha=0.2)
            ax_idx += 1

        ax_res = axes[ax_idx]
        ax_res.set_xlabel("Iteration")
        ax_res.set_ylabel("Residual")
        ax_res.set_yscale("log")
        ax_res.grid(True, alpha=0.3)
        residual_line, = ax_res.plot([], [], "r-", linewidth=2)

        self._realtime_axes = {"prediction": ax_pred, "reference": ax_ref, "residual": ax_res}
        self._realtime_handles = {
            "pred_scatter": scat_pred,
            "ref_scatter": scat_ref,
            "residual_line": residual_line,
        }

    @staticmethod
    def _set_realtime_axis_limits(ax, coords: np.ndarray) -> None:
        if ax is None or coords.size == 0:
            return
        x_min, x_max = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
        y_min, y_max = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
        x_pad = max(1e-6, 0.02 * max(x_max - x_min, 1.0))
        y_pad = max(1e-6, 0.02 * max(y_max - y_min, 1.0))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    def _finalize_animation(self, results_dir: str):
        vis_cfg = getattr(self.config, "visualization", None)
        if not vis_cfg:
            return

        if self._realtime_mode_active:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.close(self._realtime_fig)
            self._realtime_mode_active = False

        if vis_cfg.save_animation and self._anim_snapshots:
            dataset_label = "test" if self._anim_dataset_kind == "test" else "train"
            animation_path = os.path.join(results_dir, f"{dataset_label}_solution_evolution.gif")
            try:
                self.create_animation(
                    self._anim_snapshots,
                    self._anim_coords,
                    animation_path,
                    render_func=self._render_animation_frame,
                    duration=200,
                )
                print(f"  Animation saved: {animation_path}")
            except Exception as exc:
                print(f"  Warning: Failed to create animation: {exc}")
            finally:
                self._anim_snapshots = []

    def _render_animation_frame(self, coords: np.ndarray, snapshot: np.ndarray, frame_idx: int):
        fig = self._create_figure(figsize=(18, 6))
        residuals = self._anim_residuals[: frame_idx + 1]
        iterations = self._anim_iterations[: frame_idx + 1]
        vars_list = getattr(self.config, "vars_list", ["u"])

        if self.n_dim == 1:
            ax_sol = fig.add_subplot(1, 2, 1)
            coord_flat = coords[:, 0] if coords.ndim > 1 else coords
            order = np.argsort(coord_flat)
            x_sorted = coord_flat[order]
            pred_sorted = (snapshot[:, 0] if snapshot.ndim > 1 else snapshot)[order]
            ax_sol.plot(x_sorted, pred_sorted, "b-", linewidth=2, label="Prediction")
            if self._anim_reference_values is not None:
                ref_sorted = self._anim_reference_values.squeeze()[order]
                ax_sol.plot(x_sorted, ref_sorted, "r--", linewidth=2, label="Reference")
                ax_sol.legend(loc="best")
            ax_sol.set_xlabel(self.config.spatial_vars[0])
            ax_sol.set_ylabel(vars_list[0])
            ax_sol.set_title(f"Prediction (iter {iterations[-1]})")
            ax_sol.grid(True, alpha=0.3)
            ax_res = fig.add_subplot(1, 2, 2)
        else:
            ax_res = self._render_animation_frame_2d(fig, coords, snapshot, iterations, vars_list)

        ax_res.semilogy(iterations, residuals, "r-", linewidth=2)
        ax_res.scatter([iterations[-1]], [residuals[-1]], c="red", s=60, zorder=5)
        ax_res.set_xlabel("Iteration")
        ax_res.set_ylabel("Residual")
        ax_res.set_title(f"Residual History (Res={residuals[-1]:.2e})")
        ax_res.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _render_animation_frame_2d(self, fig, coords, snapshot, iterations, vars_list):
        columns = 3 if self._anim_reference_values is not None else 2
        ax_sol = fig.add_subplot(1, columns, 1)
        z_pred = snapshot[:, 0] if snapshot.ndim > 1 else snapshot
        pred_render = self._render_animation_field_2d(ax_sol, coords, z_pred, vars_list[0])
        fig.colorbar(pred_render, ax=ax_sol, label=vars_list[0])
        ax_sol.set_xlabel(self.config.spatial_vars[0])
        ax_sol.set_ylabel(self.config.spatial_vars[1])
        ax_sol.set_title(f"Prediction (iter {iterations[-1]})")
        ax_sol.grid(True, linestyle="--", alpha=0.2)

        if self._anim_reference_values is not None:
            ax_ref = fig.add_subplot(1, columns, 2)
            ref_render = self._render_animation_field_2d(
                ax_ref,
                coords,
                self._anim_reference_values.squeeze(),
                "Reference",
            )
            fig.colorbar(ref_render, ax=ax_ref, label="Reference")
            ax_ref.set_xlabel(self.config.spatial_vars[0])
            ax_ref.set_ylabel(self.config.spatial_vars[1])
            ax_ref.set_title("Reference")
            ax_ref.grid(True, linestyle="--", alpha=0.2)
            return fig.add_subplot(1, columns, 3)

        return fig.add_subplot(1, columns, 2)

    def _render_animation_field_2d(self, ax, coords: np.ndarray, values: np.ndarray, colorbar_label: str):
        grid_payload = self._build_animation_grid_payload(coords, values)
        if grid_payload is not None:
            x_mesh, y_mesh, z_grid = grid_payload
            return ax.pcolormesh(
                x_mesh,
                y_mesh,
                z_grid,
                cmap=self._anim_colormap,
                shading=self._animation_grid_shading(),
            )
        return ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap=self._anim_colormap,
            s=20,
            edgecolors="none",
        )

    def _build_animation_grid_payload(self, coords: np.ndarray, values: np.ndarray):
        plotter = getattr(self, "dimensional_visualizer", None)
        if plotter is None or coords.ndim != 2 or coords.shape[1] != 2:
            return None

        try:
            return plotter._build_structured_grid_payload(coords[:, 0], coords[:, 1], values)
        except Exception:
            return None

    def _animation_grid_shading(self) -> str:
        vis_cfg = getattr(self.config, "visualization", None)
        if vis_cfg is None:
            return "auto"
        grid_kwargs = getattr(vis_cfg, "grid_kwargs", {}) or {}
        return grid_kwargs.get("shading", "auto")
