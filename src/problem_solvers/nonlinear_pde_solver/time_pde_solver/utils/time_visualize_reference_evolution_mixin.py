import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


class TimePDEReferenceEvolutionMixin:
    def _plot_evolution_step_with_reference_1d(
        self,
        T: float,
        u: np.ndarray,
        x: np.ndarray,
        u_ref: Optional[np.ndarray] = None,
        solver=None,
        save_path: Optional[str] = None,
    ) -> None:
        x_flat = x[:, 0] if x.ndim > 1 else x
        u_flat = u[:, 0] if u.ndim > 1 else u
        reference_state = self._build_reference_state(u_ref, solver)

        if self._scatter_numerical is None:
            self._initialize_reference_plot_1d(x_flat, u_flat, reference_state)

        self._scatter_numerical.set_offsets(np.column_stack((x_flat, u_flat)))
        error_metrics = self._update_reference_line(reference_state, x_flat, u_flat)
        step_index, dt = self._advance_time_tracking(T)
        self._update_reference_plot_title(T, step_index, dt, error_metrics)
        self._update_reference_axis_limits(u_flat, reference_state, error_metrics)
        if error_metrics is not None and step_index <= 5:
            self._print_reference_error_diagnostics(step_index, T, u_flat, error_metrics)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.05)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def _build_reference_state(self, u_ref: Optional[np.ndarray], solver=None):
        if (
            u_ref is None
            or solver is None
            or not hasattr(solver, "reference_solution")
            or solver.reference_solution is None
        ):
            return None
        x_ref = solver.reference_solution["x_ref"]
        sort_idx = np.argsort(x_ref)
        return {
            "x_ref": x_ref,
            "x_sorted": x_ref[sort_idx],
            "u_sorted": u_ref[sort_idx],
            "u_ref": u_ref,
        }

    def _initialize_reference_plot_1d(
        self,
        x_flat: np.ndarray,
        u_flat: np.ndarray,
        reference_state,
    ) -> None:
        self.ax.clear()
        self.fig.subplots_adjust(top=0.88)
        self._scatter_numerical = self.ax.scatter(x_flat, u_flat, c="r", label="Numerical", s=20)

        if reference_state is not None:
            self._line_reference, = self.ax.plot(
                reference_state["x_sorted"],
                reference_state["u_sorted"],
                "b--",
                label="Reference Solution",
                linewidth=2,
                alpha=0.7,
            )
            print(
                "    Reference initialized with "
                f"{len(reference_state['x_ref'])} points, "
                f"x_range=[{reference_state['x_ref'].min():.3f}, {reference_state['x_ref'].max():.3f}]"
            )
        else:
            self._line_reference = None
            print("    No reference solution available for initialization")

        x_min, x_max, u_min, u_max = self._compute_reference_plot_extents(
            x_flat,
            u_flat,
            reference_state,
        )
        x_margin = (x_max - x_min) * 0.05
        u_margin = (u_max - u_min) * 0.1 if u_max > u_min else 0.1
        self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
        self.ax.set_ylim(u_min - u_margin, u_max + u_margin)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("u")
        self.ax.set_title("Time PDE Real-time Solution", pad=20)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def _compute_reference_plot_extents(
        self,
        x_flat: np.ndarray,
        u_flat: np.ndarray,
        reference_state,
    ) -> Tuple[float, float, float, float]:
        x_min, x_max = x_flat.min(), x_flat.max()
        u_min, u_max = u_flat.min(), u_flat.max()
        if reference_state is None:
            return x_min, x_max, u_min, u_max
        return (
            min(x_min, reference_state["x_ref"].min()),
            max(x_max, reference_state["x_ref"].max()),
            min(u_min, reference_state["u_ref"].min()),
            max(u_max, reference_state["u_ref"].max()),
        )

    def _update_reference_line(self, reference_state, x_flat: np.ndarray, u_flat: np.ndarray):
        if reference_state is None or self._line_reference is None:
            return None

        self._line_reference.set_data(reference_state["x_sorted"], reference_state["u_sorted"])
        interp_func = interp1d(
            reference_state["x_ref"],
            reference_state["u_ref"],
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        u_ref_interp = interp_func(x_flat)
        error = np.abs(u_flat - u_ref_interp)
        return {
            "u_ref_interp": u_ref_interp,
            "max_error": np.max(error),
            "l2_error": np.sqrt(np.mean(error**2)),
            "mean_error": np.mean(error),
            "std_error": np.std(error),
            "u_min": min(u_flat.min(), reference_state["u_sorted"].min()),
            "u_max": max(u_flat.max(), reference_state["u_sorted"].max()),
        }

    def _advance_time_tracking(self, T: float) -> Tuple[int, float]:
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        if not hasattr(self, "_initial_time"):
            self._initial_time = T
        if not hasattr(self, "_last_time"):
            self._last_time = T

        dt = T - self._last_time if self._step_counter > 0 else 0
        self._last_time = T
        self._step_counter += 1
        return self._step_counter, dt

    def _update_reference_plot_title(
        self,
        T: float,
        step_index: int,
        dt: float,
        error_metrics,
    ) -> None:
        title_lines = [f"Time: T = {T:.4f}s, Step: {step_index}, dt = {dt:.4f}s"]
        if error_metrics is not None:
            title_lines.append(
                f"Max Error: {error_metrics['max_error']:.2e}, L2 Error: {error_metrics['l2_error']:.2e}"
            )
        self.ax.set_title("\n".join(title_lines), pad=15, fontsize=11)

    def _update_reference_axis_limits(self, u_flat: np.ndarray, reference_state, error_metrics) -> None:
        if error_metrics is not None:
            u_min_current = error_metrics["u_min"]
            u_max_current = error_metrics["u_max"]
        elif reference_state is not None:
            u_min_current = min(u_flat.min(), reference_state["u_sorted"].min())
            u_max_current = max(u_flat.max(), reference_state["u_sorted"].max())
        else:
            u_min_current = u_flat.min()
            u_max_current = u_flat.max()

        if u_max_current > u_min_current:
            margin = (u_max_current - u_min_current) * 0.1
            self.ax.set_ylim(u_min_current - margin, u_max_current + margin)
        else:
            self.ax.set_ylim(u_min_current - 0.1, u_max_current + 0.1)

    def _print_reference_error_diagnostics(
        self,
        step_index: int,
        T: float,
        u_flat: np.ndarray,
        error_metrics,
    ) -> None:
        print(f"  详细误差诊断 - Step {step_index}, T={T:.3f}:")
        print(f"    Numerical solutionRange: [{np.min(u_flat):.6f}, {np.max(u_flat):.6f}]")
        print(
            f"    Reference solutionRange: "
            f"[{np.min(error_metrics['u_ref_interp']):.6f}, {np.max(error_metrics['u_ref_interp']):.6f}]"
        )
        print(f"    Time同步Check: Numerical solutionT={T:.6f}, Reference solutionT={T:.6f} (应该Same)")
        print(
            f"    最大误差: {error_metrics['max_error']:.2e}, "
            f"L2误差: {error_metrics['l2_error']:.2e}"
        )
        print(
            f"    误差分布: Average={error_metrics['mean_error']:.2e}, "
            f"Standard deviation={error_metrics['std_error']:.2e}"
        )
