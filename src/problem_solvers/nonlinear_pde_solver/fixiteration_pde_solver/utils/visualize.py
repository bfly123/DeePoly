import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from src.abstract_class.config.base_visualize import BaseVisualizer
from src.abstract_class.config.visualize_mixins import (
    ConvergenceMixin,
    SolutionPlotMixin,
    GridReconstructionMixin,
    AnimationMixin,
    RealtimeVisualizationMixin,
    ReferenceComparisonMixin
)
import os
from src.visualization.dimensional_plotter import DimensionalVisualizer


class FixIterationPDEVisualizer(BaseVisualizer, ConvergenceMixin, SolutionPlotMixin,
                                GridReconstructionMixin, AnimationMixin, RealtimeVisualizationMixin,
                                ReferenceComparisonMixin):

    def __init__(self, config):
        super().__init__(config)
        # 使用维度可视化器进行智能维度判断
        self.dimensional_visualizer = DimensionalVisualizer(config)
        self.n_dim = self.dimensional_visualizer.n_dim
        self.iteration_history = []
        self.residual_history = []
        self.solution_history = []
        self._scatter_solution = None
        self._line_residual = None
        self._colorbar = None
        self.fig = None
        self.ax_solution = None
        self.ax_residual = None
        self._realtime_mode_active = False

    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None):
        U = self._to_numpy_cpu(prediction)
        fig = self._create_figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        x_coords = data.get('x_segments') or data.get('x')
        if isinstance(x_coords, list):
            x_coords = np.vstack(x_coords)

        if self.n_dim == 1:
            x_flat = x_coords[:, 0] if x_coords.ndim > 1 else x_coords
            u_flat = U.flatten()
            sort_idx = np.argsort(x_flat)
            ax.plot(x_flat[sort_idx], u_flat[sort_idx], 'b-', linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.grid(True, alpha=0.3)
        else:
            Xi, Yi, Ui = self.reconstruct_2d_grid(x_coords, U)
            im = ax.pcolormesh(Xi, Yi, Ui, cmap='RdBu_r', shading='gouraud')
            plt.colorbar(im, ax=ax, label='u')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')

        ax.set_title('Solution')
        plt.tight_layout()
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def reset_animation_data(self):
        self.iteration_history = []
        self.residual_history = []
        self.solution_history = []
        print("动画数据已重置")

    def update_animation_data(self, iteration: int, residual: float, U: np.ndarray,
                            data_train: Dict, animation_skip: int = 1) -> None:
        if iteration % animation_skip == 0:
            U_cpu = self._to_numpy_cpu(U)
            self.iteration_history.append(iteration)
            self.residual_history.append(residual)
            self.solution_history.append(U_cpu.copy())

            if getattr(self.config, 'realtime_visualization', False):
                x_coords = data_train.get('x', data_train.get('x_segments', []))
                if isinstance(x_coords, list):
                    x_plot = np.vstack(x_coords) if x_coords else None
                else:
                    x_plot = x_coords

                if x_plot is not None and x_plot.size > 0 and U_cpu.size > 0:
                    self.plot_iteration_step(iteration, residual, U_cpu, x_plot)

    def plot_iteration_step(self, iteration: int, residual: float, U: np.ndarray, x: np.ndarray) -> None:
        if not self._realtime_mode_active:
            self._init_realtime_plot()

        if self.n_dim == 1:
            self._plot_iteration_step_1d(iteration, residual, U, x)
        else:
            self._plot_iteration_step_2d(iteration, residual, U, x)

    def _init_realtime_plot(self):
        if self._realtime_mode_active:
            return

        plt.ion()
        self.fig = self._create_figure(figsize=(14, 5))
        self.ax_solution = self.fig.add_subplot(121)
        self.ax_residual = self.fig.add_subplot(122)
        self.fig.tight_layout(pad=2.0)
        self._realtime_mode_active = True

    def _plot_iteration_step_1d(self, iteration: int, residual: float, U: np.ndarray, x: np.ndarray) -> None:
        if self._scatter_solution is None:
            self.ax_solution.clear()
            x_flat = x[:, 0] if x.ndim > 1 else x
            u_flat = U[:, 0] if U.ndim > 1 else U
            self._scatter_solution = self.ax_solution.scatter(x_flat, u_flat, c='b', s=20)
            self.ax_solution.set_xlabel('x')
            self.ax_solution.set_ylabel('u')
            self.ax_solution.grid(True, alpha=0.3)
        else:
            x_flat = x[:, 0] if x.ndim > 1 else x
            u_flat = U[:, 0] if U.ndim > 1 else U
            self._scatter_solution.set_offsets(np.column_stack((x_flat, u_flat)))

        u_flat = U[:, 0] if U.ndim > 1 else U
        u_min, u_max = u_flat.min(), u_flat.max()
        if u_max > u_min:
            margin = (u_max - u_min) * 0.1
            self.ax_solution.set_ylim(u_min - margin, u_max + margin)

        self.ax_solution.set_title(f'Solution at Iteration {iteration}')
        self._update_residual_plot()

    def _plot_iteration_step_2d(self, iteration: int, residual: float, U: np.ndarray, x: np.ndarray) -> None:
        self.ax_solution.clear()
        u_vals = U[:, 0] if U.ndim > 1 else U

        scatter = self.ax_solution.scatter(x[:, 0], x[:, 1], c=u_vals, cmap='viridis', s=15, alpha=0.8)
        if not hasattr(self, '_colorbar') or self._colorbar is None:
            self._colorbar = self.fig.colorbar(scatter, ax=self.ax_solution)

        self.ax_solution.set_xlabel('x')
        self.ax_solution.set_ylabel('t')
        self.ax_solution.set_title(f'Solution at Iteration {iteration}')
        self._update_residual_plot()

    def _update_residual_plot(self):
        if self._line_residual is None:
            self._line_residual, = self.ax_residual.semilogy(
                self.iteration_history, self.residual_history, 'r-', linewidth=2
            )
            self.ax_residual.set_xlabel('Iteration')
            self.ax_residual.set_ylabel('Residual')
            self.ax_residual.set_title('Residual History')
            self.ax_residual.grid(True, alpha=0.3)
        else:
            self._line_residual.set_data(self.iteration_history, self.residual_history)
            if len(self.iteration_history) > 0:
                self.ax_residual.set_xlim(0, max(self.iteration_history) * 1.1)
            if len(self.residual_history) > 0:
                y_min = min(self.residual_history)
                y_max = max(self.residual_history)
                if y_max > y_min:
                    self.ax_residual.set_ylim(y_min * 0.5, y_max * 2.0)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def create_iteration_gif(self, filename: str = "iteration_evolution.gif") -> None:
        if not self.iteration_history or not self.solution_history:
            return

        snapshots = []
        for iteration, residual, solution in zip(self.iteration_history, self.residual_history, self.solution_history):
            if hasattr(self.config, 'x_domain'):
                n_points = len(solution)
                x = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], n_points)
                x = x.reshape(-1, 1)
                snapshots.append(solution)

        self.create_animation(snapshots, x, filename, duration=200,
                            title_func=lambda i: f'Iteration {self.iteration_history[i]}, Residual: {self.residual_history[i]:.2e}')

    def finalize_realtime_animation(self) -> None:
        if self._realtime_mode_active:
            plt.ioff()
            self._realtime_mode_active = False

    def plot_final_solution(self, U: np.ndarray, data: Dict, save_path: Optional[str] = None) -> None:
        U = self._to_numpy_cpu(U)
        fig = self._create_figure(figsize=(12, 5))

        if self.n_dim == 1:
            x_coords = data.get('x_segments') or data.get('x')
            if isinstance(x_coords, list):
                x_coords = np.vstack(x_coords)
            if x_coords is None:
                x_coords = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], len(U))

            x_flat = x_coords[:, 0] if x_coords.ndim > 1 else x_coords
            u_flat = U.flatten()
            sort_idx = np.argsort(x_flat)

            ax = fig.add_subplot(111)
            ax.plot(x_flat[sort_idx], u_flat[sort_idx], 'b-', linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title('Final Steady-State Solution')
            ax.grid(True, alpha=0.3)
        else:
            x_coords = data.get('x_segments') or data.get('x')
            if isinstance(x_coords, list):
                x_coords = np.vstack(x_coords)

            Xi, Yi, Ui = self.reconstruct_2d_grid(x_coords, U)
            ax = fig.add_subplot(111)
            im = ax.pcolormesh(Xi, Yi, Ui, cmap='RdBu_r', shading='gouraud')
            plt.colorbar(im, ax=ax, label='u')
            ax.set_xlabel('x')
            ax.set_ylabel(getattr(self.config, 'spatial_vars', ['x', 'y'])[1])
            ax.set_title('Final Steady-State Solution')
            ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def close_evolution_plot(self) -> None:
        if self._realtime_mode_active:
            plt.ioff()
            self._realtime_mode_active = False
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def save_convergence_history(self, conv_info: Dict, save_dir: str) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].semilogy(conv_info["residual_history"])
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Residual")
        axes[0, 0].set_title("Convergence History")
        axes[0, 0].grid(True, alpha=0.3)

        if "dt_history" in conv_info:
            axes[0, 1].plot(conv_info["dt_history"])
            axes[0, 1].set_xlabel("Iteration")
            axes[0, 1].set_ylabel("dt (pseudo-time step)")
            axes[0, 1].set_title("Adaptive Time Step")
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, "No time stepping", ha='center', va='center')

        if "solution_norm_history" in conv_info:
            axes[1, 0].plot(conv_info["solution_norm_history"])
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("||U||")
            axes[1, 0].set_title("Solution Norm Evolution")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, "No norm history", ha='center', va='center')

        if len(conv_info["residual_history"]) > 1:
            residuals = np.array(conv_info["residual_history"])
            rate = -np.diff(np.log(residuals + 1e-16))
            axes[1, 1].plot(rate)
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Convergence Rate")
            axes[1, 1].set_title("Local Convergence Rate")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, "convergence_history.png")
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def visualize_solution(self, U_steady: np.ndarray, data: Dict, save_dir: str) -> None:
        U_steady = self._to_numpy_cpu(U_steady)
        x = data['x']
        n_eqs = U_steady.shape[1] if U_steady.ndim > 1 else 1

        if self.n_dim == 1:
            fig, axes = plt.subplots(1, n_eqs, figsize=(6*n_eqs, 5))
            if n_eqs == 1:
                axes = [axes]

            for eq_idx in range(n_eqs):
                u_vals = U_steady[:, eq_idx] if U_steady.ndim > 1 else U_steady
                x_flat = x[:, 0] if x.ndim > 1 else x
                sort_idx = np.argsort(x_flat)

                axes[eq_idx].plot(x_flat[sort_idx], u_vals[sort_idx], 'b-', linewidth=2)
                axes[eq_idx].set_xlabel('x')
                axes[eq_idx].set_ylabel(f'u_{eq_idx}')
                axes[eq_idx].set_title(f'Steady-State Solution (Equation {eq_idx})')
                axes[eq_idx].grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(save_dir, "steady_state_solution.png")
            self._save_figure(fig, save_path)
            self._close_figure(fig)
        else:
            Xi, Yi, _ = self.reconstruct_2d_grid(x, U_steady[:, 0:1] if U_steady.ndim > 1 else U_steady)
            for eq_idx in range(n_eqs):
                fig = self._create_figure(figsize=(10, 8))
                ax = fig.add_subplot(111)
                u_vals = U_steady[:, eq_idx] if U_steady.ndim > 1 else U_steady
                _, _, Ui = self.reconstruct_2d_grid(x, u_vals)

                im = ax.pcolormesh(Xi, Yi, Ui, cmap='RdBu_r', shading='gouraud')
                plt.colorbar(im, ax=ax, label=f'u_{eq_idx}')
                ax.set_xlabel('x')
                ax.set_ylabel(getattr(self.config, 'spatial_vars', ['x', 'y'])[1])
                ax.set_title(f'Steady-State Solution (Equation {eq_idx})')
                ax.set_aspect('equal')
                plt.tight_layout()

                save_path = os.path.join(save_dir, f"steady_state_solution_eq{eq_idx}.png")
                self._save_figure(fig, save_path)
                self._close_figure(fig)

    def save_solution_data(self, U_train: np.ndarray, U_test: np.ndarray,
                           coeffs_steady: np.ndarray, conv_info: Dict,
                           data_train: Dict, data_test: Dict, save_dir: str) -> None:
        U_train = self._to_numpy_cpu(U_train)
        U_test = self._to_numpy_cpu(U_test)
        coeffs_steady = self._to_numpy_cpu(coeffs_steady)

        save_path = os.path.join(save_dir, "solution_data.npz")
        np.savez(
            save_path,
            x_train=data_train['x'],
            x_test=data_test['x'],
            U_train=U_train,
            U_test=U_test,
            coeffs=coeffs_steady,
            residual_history=conv_info["residual_history"],
            dt_history=conv_info.get("dt_history", []),
            iterations=conv_info["iterations"],
            converged=conv_info["converged"]
        )
        print(f"✓ Solution data saved: {save_path}")
