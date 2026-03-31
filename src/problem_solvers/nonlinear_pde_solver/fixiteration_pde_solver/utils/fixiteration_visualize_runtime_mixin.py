import numpy as np
import matplotlib.pyplot as plt


class FixIterationVisualizeRuntimeMixin:
    def reset_animation_data(self):
        self.iteration_history = []
        self.residual_history = []
        self.solution_history = []
        print("动画数据已重置")

    def update_animation_data(
        self,
        iteration: int,
        residual: float,
        U: np.ndarray,
        data_train,
        animation_skip: int = 1,
    ) -> None:
        if iteration % animation_skip != 0:
            return
        U_cpu = self._to_numpy_cpu(U)
        self.iteration_history.append(iteration)
        self.residual_history.append(residual)
        self.solution_history.append(U_cpu.copy())
        if getattr(self.config, "realtime_visualization", False):
            x_plot = self._extract_coords({"x": data_train.get("x"), "x_segments": data_train.get("x_segments", [])})
            if x_plot is not None and x_plot.size > 0 and U_cpu.size > 0:
                self.plot_iteration_step(iteration, residual, U_cpu, x_plot)

    def plot_iteration_step(self, iteration: int, residual: float, U: np.ndarray, x: np.ndarray) -> None:
        if not self._realtime_mode_active:
            self._init_realtime_plot()
        if self.n_dim == 1:
            self._plot_iteration_step_1d(iteration, U, x)
        else:
            self._plot_iteration_step_2d(iteration, U, x)
        self._update_residual_plot()

    def _init_realtime_plot(self):
        if self._realtime_mode_active:
            return
        plt.ion()
        self.fig = self._create_figure(figsize=(14, 5))
        self.ax_solution = self.fig.add_subplot(121)
        self.ax_residual = self.fig.add_subplot(122)
        self.fig.tight_layout(pad=2.0)
        self._realtime_mode_active = True

    def _plot_iteration_step_1d(self, iteration: int, U: np.ndarray, x: np.ndarray) -> None:
        x_flat = x[:, 0] if x.ndim > 1 else x
        u_flat = U[:, 0] if U.ndim > 1 else U
        if self._scatter_solution is None:
            self.ax_solution.clear()
            self._scatter_solution = self.ax_solution.scatter(x_flat, u_flat, c="b", s=20)
            self.ax_solution.set_xlabel("x")
            self.ax_solution.set_ylabel("u")
            self.ax_solution.grid(True, alpha=0.3)
        else:
            self._scatter_solution.set_offsets(np.column_stack((x_flat, u_flat)))
        u_min, u_max = u_flat.min(), u_flat.max()
        if u_max > u_min:
            margin = (u_max - u_min) * 0.1
            self.ax_solution.set_ylim(u_min - margin, u_max + margin)
        self.ax_solution.set_title(f"Solution at Iteration {iteration}")

    def _plot_iteration_step_2d(self, iteration: int, U: np.ndarray, x: np.ndarray) -> None:
        self.ax_solution.clear()
        u_vals = U[:, 0] if U.ndim > 1 else U
        scatter = self.ax_solution.scatter(x[:, 0], x[:, 1], c=u_vals, cmap="viridis", s=15, alpha=0.8)
        if self._colorbar is None:
            self._colorbar = self.fig.colorbar(scatter, ax=self.ax_solution)
        self.ax_solution.set_xlabel("x")
        self.ax_solution.set_ylabel("t")
        self.ax_solution.set_title(f"Solution at Iteration {iteration}")

    def _update_residual_plot(self):
        if self._line_residual is None:
            self._line_residual, = self.ax_residual.semilogy(
                self.iteration_history,
                self.residual_history,
                "r-",
                linewidth=2,
            )
            self.ax_residual.set_xlabel("Iteration")
            self.ax_residual.set_ylabel("Residual")
            self.ax_residual.set_title("Residual History")
            self.ax_residual.grid(True, alpha=0.3)
        else:
            self._line_residual.set_data(self.iteration_history, self.residual_history)
            if self.iteration_history:
                self.ax_residual.set_xlim(0, max(self.iteration_history) * 1.1)
            if self.residual_history:
                y_min = min(self.residual_history)
                y_max = max(self.residual_history)
                if y_max > y_min:
                    self.ax_residual.set_ylim(y_min * 0.5, y_max * 2.0)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def create_iteration_gif(self, filename: str = "iteration_evolution.gif") -> None:
        if not self.iteration_history or not self.solution_history or not hasattr(self.config, "x_domain"):
            return
        x = self._build_animation_coords(self.solution_history[0])
        snapshots = [solution for solution in self.solution_history]
        self.create_animation(
            snapshots,
            x,
            filename,
            duration=200,
            title_func=lambda i: (
                f"Iteration {self.iteration_history[i]}, Residual: {self.residual_history[i]:.2e}"
            ),
        )

    def _build_animation_coords(self, solution) -> np.ndarray:
        n_points = len(solution)
        x = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], n_points)
        return x.reshape(-1, 1)

    def finalize_realtime_animation(self) -> None:
        if self._realtime_mode_active:
            plt.ioff()
            self._realtime_mode_active = False

    def close_evolution_plot(self) -> None:
        if self._realtime_mode_active:
            plt.ioff()
            self._realtime_mode_active = False
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
