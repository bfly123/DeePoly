import io
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interp1d


class TimePDEAnimationMixin:
    def update_animation_data(
        self,
        it: int,
        T: float,
        data_train: Dict,
        model: torch.nn.Module,
        coeffs: np.ndarray,
        fitter,
        solver=None,
        U_direct: np.ndarray = None,
        U_seg_direct: list = None,
    ) -> None:
        """UpdateanimationData - SupportReference solution对比"""
        _ = (model, U_seg_direct)
        if not self._should_update_animation_frame(it, T):
            return

        data_plot = self._resolve_plot_dataset(data_train, solver)
        self.time_history_viz.append(T)
        try:
            x_plot = self._extract_animation_coordinates(data_plot)
            if x_plot is None:
                return
            U_plot = self._resolve_animation_solution(data_plot, model, coeffs, fitter, U_direct, T)
            if U_plot is None:
                return
            self.solution_history_viz.append(np.asarray(U_plot, dtype=np.float32).copy())
            self.coeffs_history_viz.append(None if coeffs is None else np.asarray(coeffs, dtype=np.float64).copy())
            self._maybe_render_realtime_solution(T, U_plot, x_plot, solver)
            print(f"  animationData已Update: T = {T:.6f}, 总帧数 = {len(self.solution_history_viz)}")
        except Exception as exc:
            print(f"  Warning: animationDataUpdateFail: {exc}")

    def _should_update_animation_frame(self, it: int, T: float) -> bool:
        animation_skip = getattr(self.config, "animation_skip", 10)
        return it % animation_skip == 0 or T >= getattr(self.config, "T", float("inf"))

    def _resolve_plot_dataset(self, data_train: Dict, solver=None) -> Dict:
        _ = solver
        return data_train

    def _extract_animation_coordinates(self, data_train: Dict):
        if data_train is None:
            print("  Warning: data_train is None, skipping animation update")
            return None
        if "x" not in data_train and "x_segments" not in data_train:
            print("  Warning: No coordinate data in data_train, skipping animation update")
            return None
        x_coords = data_train.get("x", data_train.get("x_segments", []))
        if not hasattr(x_coords, "__iter__") or len(x_coords) == 0:
            print("  Warning: coordinate data empty, skipping animation update")
            return None
        if isinstance(x_coords, list):
            return np.vstack(x_coords) if x_coords else np.array([[0]])
        return x_coords

    def _resolve_animation_solution(
        self,
        data_plot: Dict,
        model: torch.nn.Module,
        coeffs: np.ndarray,
        fitter,
        U_direct: np.ndarray,
        T: float,
    ):
        if U_direct is not None and data_plot is getattr(fitter, "data", None):
            return U_direct
        if coeffs is None:
            print(f"  Warning: coeffs is None at T={T:.4f}, skipping animation update")
            return None
        U_plot, _ = fitter.construct(data_plot, model, coeffs)
        print("  备用Plan：Using construct ConstructSolution")
        if U_plot is None:
            print("  Warning: U_train is None, skipping animation update")
        return U_plot

    def _maybe_render_realtime_solution(
        self,
        T: float,
        U_train: np.ndarray,
        x_plot: np.ndarray,
        solver=None,
    ) -> None:
        if not getattr(self.config, "realtime_visualization", False):
            print(
                "  Real-time visualization disabled "
                f"(setting: {getattr(self.config, 'realtime_visualization', 'Not found')})"
            )
            return
        print(f"  Real-time visualization enabled, plotting at T = {T:.4f}")
        u_ref = None
        if solver and hasattr(solver, "get_reference_solution_at_time"):
            u_ref = solver.get_reference_solution_at_time(T)
            print(f"    Reference solution at T={T:.4f}: {'Yes' if u_ref is not None else 'No'}")
            if u_ref is not None:
                print(f"    Reference solution range: [{np.min(u_ref):.4f}, {np.max(u_ref):.4f}]")
        self.plot_evolution_step_with_reference(T, U_train, x_plot, u_ref, solver)
        print(f"    Real-time plot updated for T = {T:.4f}")

    def get_animation_data(self):
        """GetanimationData - convert deque to list for compatibility"""
        return list(self.time_history_viz), list(self.solution_history_viz)

    def get_animation_coeffs(self):
        """Get coefficient history aligned with animation snapshots."""
        return list(self.coeffs_history_viz)

    def save_animation(self, filename: str = "animation.gif", duration: int = 200) -> None:
        """将捕获的帧Save为GIFanimation"""
        if len(self._frames) == 0:
            print("没有帧可Save")
            return
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        self._frames[0].save(
            filename,
            save_all=True,
            append_images=self._frames[1:],
            optimize=True,
            duration=duration,
            loop=0,
        )
        print(f"GIFanimation已Save至 {filename}")
        self._frames = []

    def create_time_evolution_gif(
        self,
        time_history: List[float],
        solution_history: List[np.ndarray],
        data_train: Dict,
        filename: str = "time_evolution.gif",
        solver=None,
    ) -> None:
        """CreateTime演化GIFanimation，SupportReference solution对比"""
        if not time_history or not solution_history:
            print("Warning: 没有Time演化Data可Createanimation")
            return

        try:
            print(f"正AtCreateTime演化animation: {filename}")
            x_coords = self._get_gif_coordinates(data_train)
            if x_coords is None:
                return
            self._frames = []
            for index, (T, solution) in enumerate(zip(time_history, solution_history)):
                self._frames.append(
                    self._build_animation_frame(
                        index,
                        T,
                        solution,
                        x_coords,
                        time_history,
                        solution_history,
                        solver,
                    )
                )
                if (index + 1) % 10 == 0:
                    print(f"  已Generate {index + 1}/{len(time_history)} 帧")
            self.save_animation(filename, duration=200)
            print(f"Time演化animationCreateComplete: {filename}")
        except Exception as exc:
            print(f"animationCreateFail: {exc}")

    def _get_gif_coordinates(self, data_train: Dict):
        if "x_segments" in data_train:
            return np.vstack(data_train["x_segments"])
        if "x" in data_train:
            return data_train["x"]
        print("Warning: 无法GetxcoordinateData")
        return None

    def _build_animation_frame(
        self,
        index: int,
        T: float,
        solution: np.ndarray,
        x_coords: np.ndarray,
        time_history: List[float],
        solution_history: List[np.ndarray],
        solver=None,
    ):
        figsize = {1: (10, 6), 2: (10, 8)}.get(self.n_dim, (10, 6))
        temp_fig = plt.figure(figsize=figsize)
        temp_fig.subplots_adjust(top=0.88)
        if self.n_dim == 1:
            ax = temp_fig.add_subplot(111)
            self._build_1d_animation_frame(
                ax,
                index,
                T,
                solution,
                x_coords,
                time_history,
                solution_history,
                solver,
            )
        else:
            ax = temp_fig.add_subplot(111, projection="3d")
            self._build_2d_animation_frame(ax, T, solution, x_coords)
        plt.tight_layout()
        frame = self._capture_animation_frame(temp_fig)
        plt.close(temp_fig)
        return frame

    def _build_1d_animation_frame(
        self,
        ax,
        index: int,
        T: float,
        solution: np.ndarray,
        x_coords: np.ndarray,
        time_history: List[float],
        solution_history: List[np.ndarray],
        solver=None,
    ) -> None:
        x_flat = x_coords[:, 0] if x_coords.ndim > 1 else x_coords
        u_flat = solution.flatten()
        ax.scatter(x_flat, u_flat, c="r", label="Numerical", s=20)
        error_metrics = self._add_reference_to_animation_frame(ax, x_flat, u_flat, T, solver)
        self._configure_1d_animation_axes(ax, x_flat, u_flat, time_history, index, solution_history, solver)

        dt = time_history[index] - time_history[index - 1] if index > 0 else (
            time_history[1] - time_history[0] if len(time_history) > 1 else 0
        )
        title_text = f"Time: T = {T:.4f}s, Step: {index + 1}, dt = {dt:.4f}s"
        if error_metrics is not None:
            title_text += (
                f"\nMax Error: {error_metrics[0]:.2e}, "
                f"L2 Error: {error_metrics[1]:.2e}"
            )
        ax.set_title(title_text, fontsize=11, fontweight="bold", pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    def _add_reference_to_animation_frame(
        self,
        ax,
        x_flat: np.ndarray,
        u_flat: np.ndarray,
        T: float,
        solver=None,
    ) -> Optional[Tuple[float, float]]:
        if not (solver and hasattr(solver, "get_reference_solution_at_time")):
            return None
        u_ref = solver.get_reference_solution_at_time(T)
        if u_ref is None or not hasattr(solver, "reference_solution") or solver.reference_solution is None:
            return None
        x_ref = solver.reference_solution["x_ref"]
        sort_idx = np.argsort(x_ref)
        x_ref_sorted = x_ref[sort_idx]
        u_ref_sorted = u_ref[sort_idx]
        ax.plot(x_ref_sorted, u_ref_sorted, "b--", linewidth=2, alpha=0.7, label="Reference Solution")
        try:
            interp_func = interp1d(x_ref, u_ref, kind="cubic", bounds_error=False, fill_value="extrapolate")
            u_ref_interp = interp_func(x_flat)
            error = np.abs(u_flat - u_ref_interp)
            return np.max(error), np.sqrt(np.mean(error**2))
        except Exception as exc:
            print(f"Reference solutionInterpolationFail: {exc}")
            return None

    def _configure_1d_animation_axes(
        self,
        ax,
        x_flat: np.ndarray,
        u_flat: np.ndarray,
        time_history: List[float],
        index: int,
        solution_history: List[np.ndarray],
        solver=None,
    ) -> None:
        _ = (time_history, index)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("u", fontsize=12)
        all_solutions = np.concatenate([solution.flatten() for solution in solution_history])
        y_min, y_max = all_solutions.min(), all_solutions.max()
        if solver and hasattr(solver, "reference_solution") and solver.reference_solution is not None:
            try:
                u_ref_all = solver.reference_solution["u_ref"]
                y_min = min(y_min, u_ref_all.min())
                y_max = max(y_max, u_ref_all.max())
            except Exception:
                pass
        margin = (y_max - y_min) * 0.1
        ax.set_ylim([y_min - margin, y_max + margin])

        x_min, x_max = x_flat.min(), x_flat.max()
        if solver and hasattr(solver, "reference_solution") and solver.reference_solution is not None:
            try:
                x_ref = solver.reference_solution["x_ref"]
                x_min = min(x_min, x_ref.min())
                x_max = max(x_max, x_ref.max())
            except Exception:
                pass
        x_margin = (x_max - x_min) * 0.05
        ax.set_xlim([x_min - x_margin, x_max + x_margin])

    def _build_2d_animation_frame(self, ax, T: float, solution: np.ndarray, x_coords: np.ndarray) -> None:
        u_vals = solution.flatten()
        ax.scatter(x_coords[:, 0], x_coords[:, 1], u_vals, c=u_vals, cmap="RdBu", s=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.set_title(f"T = {T:.4f}")

    def _capture_animation_frame(self, temp_fig) -> Image.Image:
        buf = io.BytesIO()
        temp_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        return Image.open(buf)
