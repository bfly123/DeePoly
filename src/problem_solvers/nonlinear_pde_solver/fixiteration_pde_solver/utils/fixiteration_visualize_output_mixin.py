import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


class FixIterationVisualizeOutputMixin:
    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None):
        values = self._to_numpy_cpu(prediction)
        fig = self._create_figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        x_coords = self._extract_coords(data)
        if self.n_dim == 1:
            self._plot_1d_solution(ax, x_coords, values, title="Solution")
        else:
            self._plot_2d_solution(ax, x_coords, values, title="Solution")
        plt.tight_layout()
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def plot_final_solution(self, U: np.ndarray, data: Dict, save_path: Optional[str] = None) -> None:
        values = self._to_numpy_cpu(U)
        fig = self._create_figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        if self.n_dim == 1:
            x_coords = self._extract_coords(data)
            if x_coords is None:
                x_coords = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], len(values))
            self._plot_1d_solution(ax, x_coords, values, title="Final Steady-State Solution")
        else:
            self._plot_2d_solution(
                ax,
                self._extract_coords(data),
                values,
                title="Final Steady-State Solution",
            )
        plt.tight_layout()
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _extract_coords(self, data: Dict):
        x_coords = data.get("x_segments") or data.get("x")
        if isinstance(x_coords, list):
            return np.vstack(x_coords) if x_coords else None
        return x_coords

    def _plot_1d_solution(self, ax, x_coords, values, title: str) -> None:
        x_flat = x_coords[:, 0] if getattr(x_coords, "ndim", 1) > 1 else x_coords
        u_flat = values.flatten()
        sort_idx = np.argsort(x_flat)
        ax.plot(x_flat[sort_idx], u_flat[sort_idx], "b-", linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _plot_2d_solution(self, ax, x_coords, values, title: str) -> None:
        Xi, Yi, Ui = self.reconstruct_2d_grid(x_coords, values)
        im = ax.pcolormesh(Xi, Yi, Ui, cmap="RdBu_r", shading="gouraud")
        plt.colorbar(im, ax=ax, label="u")
        ax.set_xlabel("x")
        ax.set_ylabel(getattr(self.config, "spatial_vars", ["x", "y"])[1])
        ax.set_title(title)
        ax.set_aspect("equal")

    def save_convergence_history(self, conv_info: Dict, save_dir: str) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        self._plot_residual_history_panel(axes[0, 0], conv_info["residual_history"])
        self._plot_dt_history_panel(axes[0, 1], conv_info.get("dt_history"))
        self._plot_norm_history_panel(axes[1, 0], conv_info.get("solution_norm_history"))
        self._plot_convergence_rate_panel(axes[1, 1], conv_info["residual_history"])
        plt.tight_layout()
        self._save_figure(fig, os.path.join(save_dir, "convergence_history.png"))
        self._close_figure(fig)

    def _plot_residual_history_panel(self, ax, residual_history) -> None:
        ax.semilogy(residual_history)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        ax.set_title("Convergence History")
        ax.grid(True, alpha=0.3)

    def _plot_dt_history_panel(self, ax, dt_history) -> None:
        if dt_history:
            ax.plot(dt_history)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("dt (pseudo-time step)")
            ax.set_title("Adaptive Time Step")
            ax.grid(True, alpha=0.3)
            return
        ax.text(0.5, 0.5, "No time stepping", ha="center", va="center")

    def _plot_norm_history_panel(self, ax, solution_norm_history) -> None:
        if solution_norm_history:
            ax.plot(solution_norm_history)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("||U||")
            ax.set_title("Solution Norm Evolution")
            ax.grid(True, alpha=0.3)
            return
        ax.text(0.5, 0.5, "No norm history", ha="center", va="center")

    def _plot_convergence_rate_panel(self, ax, residual_history) -> None:
        if len(residual_history) <= 1:
            return
        residuals = np.array(residual_history)
        rate = -np.diff(np.log(residuals + 1e-16))
        ax.plot(rate)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Convergence Rate")
        ax.set_title("Local Convergence Rate")
        ax.grid(True, alpha=0.3)

    def visualize_solution(self, U_steady: np.ndarray, data: Dict, save_dir: str) -> None:
        values = self._to_numpy_cpu(U_steady)
        x_coords = data["x"]
        n_eqs = values.shape[1] if values.ndim > 1 else 1
        if self.n_dim == 1:
            self._visualize_1d_solution_set(values, x_coords, n_eqs, save_dir)
            return
        self._visualize_2d_solution_set(values, x_coords, n_eqs, save_dir)

    def _visualize_1d_solution_set(self, values, x_coords, n_eqs: int, save_dir: str) -> None:
        fig, axes = plt.subplots(1, n_eqs, figsize=(6 * n_eqs, 5))
        if n_eqs == 1:
            axes = [axes]
        for eq_idx in range(n_eqs):
            eq_values = values[:, eq_idx] if values.ndim > 1 else values
            self._plot_1d_equation_solution(axes[eq_idx], x_coords, eq_values, eq_idx)
        plt.tight_layout()
        self._save_figure(fig, os.path.join(save_dir, "steady_state_solution.png"))
        self._close_figure(fig)

    def _plot_1d_equation_solution(self, ax, x_coords, values, eq_idx: int) -> None:
        x_flat = x_coords[:, 0] if x_coords.ndim > 1 else x_coords
        sort_idx = np.argsort(x_flat)
        ax.plot(x_flat[sort_idx], values[sort_idx], "b-", linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel(f"u_{eq_idx}")
        ax.set_title(f"Steady-State Solution (Equation {eq_idx})")
        ax.grid(True, alpha=0.3)

    def _visualize_2d_solution_set(self, values, x_coords, n_eqs: int, save_dir: str) -> None:
        Xi, Yi, _ = self.reconstruct_2d_grid(x_coords, values[:, 0:1] if values.ndim > 1 else values)
        for eq_idx in range(n_eqs):
            fig = self._create_figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            eq_values = values[:, eq_idx] if values.ndim > 1 else values
            _, _, Ui = self.reconstruct_2d_grid(x_coords, eq_values)
            im = ax.pcolormesh(Xi, Yi, Ui, cmap="RdBu_r", shading="gouraud")
            plt.colorbar(im, ax=ax, label=f"u_{eq_idx}")
            ax.set_xlabel("x")
            ax.set_ylabel(getattr(self.config, "spatial_vars", ["x", "y"])[1])
            ax.set_title(f"Steady-State Solution (Equation {eq_idx})")
            ax.set_aspect("equal")
            plt.tight_layout()
            self._save_figure(fig, os.path.join(save_dir, f"steady_state_solution_eq{eq_idx}.png"))
            self._close_figure(fig)

    def save_solution_data(
        self,
        U_train: np.ndarray,
        U_spotter_test: np.ndarray,
        U_test: np.ndarray,
        coeffs_steady: np.ndarray,
        conv_info: Dict,
        data_train: Dict,
        data_test: Dict,
        save_dir: str,
    ) -> None:
        save_path = os.path.join(save_dir, "solution_data.npz")
        np.savez(
            save_path,
            x_train=data_train["x"],
            x_test=data_test["x"],
            U_train=self._to_numpy_cpu(U_train),
            spotter_test=self._to_numpy_cpu(U_spotter_test),
            sniper_test=self._to_numpy_cpu(U_test),
            U_test=self._to_numpy_cpu(U_test),
            coeffs=self._to_numpy_cpu(coeffs_steady),
            residual_history=conv_info["residual_history"],
            dt_history=conv_info.get("dt_history", []),
            iterations=conv_info["iterations"],
            converged=conv_info["converged"],
        )
        print(f"✓ Solution data saved: {save_path}")
