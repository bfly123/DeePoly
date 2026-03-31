from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class TimePDESolutionMixin:
    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """Unified solution plotting - eliminate dimension-based branching"""
        handler = {
            1: self._plot_solution_1d,
            2: self._plot_solution_2d,
        }.get(self.n_dim, self._plot_solution_1d)
        handler(data, prediction, save_path)

    def _plot_solution_1d(
        self,
        data: Dict,
        prediction: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """1DSolution的Visualization"""
        fig = self._create_figure(figsize=(14, 5))
        x_global = np.vstack(data["x_segments"])
        u_global = np.vstack(data["u_segments"])
        x = x_global[:, 0]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        u_sorted = u_global.flatten()[sort_idx]
        pred_sorted = prediction.flatten()[sort_idx]

        axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
        self._plot_1d_solution_panels(axes, x_sorted, u_sorted, pred_sorted)
        self._add_segment_boundaries_1d(axes)
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _plot_1d_solution_panels(
        self,
        axes,
        x_sorted: np.ndarray,
        u_sorted: np.ndarray,
        pred_sorted: np.ndarray,
    ) -> None:
        ax1, ax2, ax3 = axes
        ax1.plot(x_sorted, u_sorted, "b-", linewidth=2, label="True实Solution")
        ax1.set_title("True实Solution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(x)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(x_sorted, pred_sorted, "r-", linewidth=2, label="PredictionSolution")
        ax2.set_title("PredictionSolution")
        ax2.set_xlabel("x")
        ax2.set_ylabel("u(x)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3.plot(x_sorted, u_sorted, "b-", linewidth=2, label="True实Solution", alpha=0.7)
        ax3.plot(x_sorted, pred_sorted, "r--", linewidth=2, label="PredictionSolution", alpha=0.7)
        ax3.set_title("对比")
        ax3.set_xlabel("x")
        ax3.set_ylabel("u(x)")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    def _add_segment_boundaries_1d(self, axes: List) -> None:
        if not (hasattr(self.config, "n_segments") and hasattr(self.config, "x_domain")):
            return
        x_min, x_max = self.config.x_domain[0]
        n_seg = (
            self.config.n_segments[0]
            if isinstance(self.config.n_segments, (list, tuple))
            else self.config.n_segments
        )
        segment_boundaries = np.linspace(x_min, x_max, n_seg + 1)
        for ax in axes:
            for boundary in segment_boundaries[1:-1]:
                ax.axvline(boundary, color="k", linestyle=":", alpha=0.5, linewidth=1)

    def _plot_solution_2d(
        self,
        data: Dict,
        prediction: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """2DSolution的Visualization"""
        fig = self._create_figure(figsize=(12, 5))
        x_global = np.vstack(data["x_segments"])
        u_global = np.vstack(data["u_segments"])
        x = np.unique(x_global[:, 0])
        y = np.unique(x_global[:, 1])
        X, Y = np.meshgrid(x, y)
        pred_grid = prediction.reshape(len(y), len(x))
        u_grid = u_global.reshape(len(y), len(x))

        ax1 = fig.add_subplot(121)
        im1 = ax1.pcolormesh(X, Y, u_grid, cmap="viridis")
        plt.colorbar(im1, ax=ax1)
        ax1.set_title("True实Solution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")

        ax2 = fig.add_subplot(122)
        im2 = ax2.pcolormesh(X, Y, pred_grid, cmap="viridis")
        plt.colorbar(im2, ax=ax2)
        ax2.set_title("PredictionSolution")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_aspect("equal")

        self._add_segment_boundaries_2d([ax1, ax2])
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def plot_error(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """绘制误差分布 - According toDimensions自动选择1D或2DVisualization"""
        handler = self._plot_error_1d if self.n_dim == 1 else self._plot_error_2d
        handler(data, prediction, save_path)

    def _plot_error_1d(
        self,
        data: Dict,
        prediction: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """1D误差分布Visualization"""
        fig = self._create_figure(figsize=(12, 8))
        x_global = np.vstack(data["x_segments"])
        u_global = np.vstack(data["u_segments"])
        error = np.abs(prediction - u_global)
        relative_error = error / (np.abs(u_global) + 1e-12)

        x = x_global[:, 0]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        error_sorted = error.flatten()[sort_idx]
        rel_error_sorted = relative_error.flatten()[sort_idx]
        u_sorted = u_global.flatten()[sort_idx]
        pred_sorted = prediction.flatten()[sort_idx]

        ax1 = fig.add_subplot(221)
        ax1.plot(x_sorted, error_sorted, "r-", linewidth=2, label="Absolute Error")
        ax1.set_title("Absolute Error Distribution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("|u_true - u_pred|")
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        ax2 = fig.add_subplot(222)
        ax2.plot(x_sorted, rel_error_sorted, "g-", linewidth=2, label="Relative Error")
        ax2.set_title("Relative Error Distribution")
        ax2.set_xlabel("x")
        ax2.set_ylabel("|u_true - u_pred|/|u_true|")
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        ax3 = fig.add_subplot(223)
        ax3.semilogy(x_sorted, np.abs(u_sorted), "b-", linewidth=2, label="|True实Solution|", alpha=0.7)
        ax3.semilogy(
            x_sorted,
            np.abs(pred_sorted),
            "r--",
            linewidth=2,
            label="|PredictionSolution|",
            alpha=0.7,
        )
        ax3.set_title("Solution的量级对比")
        ax3.set_xlabel("x")
        ax3.set_ylabel("|u|")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        ax4 = fig.add_subplot(224)
        ax4.hist(np.log10(error_sorted + 1e-16), bins=30, alpha=0.7, edgecolor="black")
        ax4.set_title("Error Distribution Histogram")
        ax4.set_xlabel("log10(Absolute Error)")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)

        self._add_segment_boundaries_1d([ax1, ax2, ax3])
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _plot_error_2d(
        self,
        data: Dict,
        prediction: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """2D误差分布Visualization"""
        fig = self._create_figure(figsize=(15, 5))
        x_global = np.vstack(data["x_segments"])
        u_global = np.vstack(data["u_segments"])
        error = np.abs(prediction - u_global)
        relative_error = error / (np.abs(u_global) + 1e-12)

        x = np.unique(x_global[:, 0])
        y = np.unique(x_global[:, 1])
        X, Y = np.meshgrid(x, y)
        error_grid = error.reshape(len(y), len(x))
        rel_error_grid = relative_error.reshape(len(y), len(x))

        ax1 = fig.add_subplot(131)
        im1 = ax1.pcolormesh(X, Y, error_grid, cmap="hot")
        plt.colorbar(im1, ax=ax1, label="Absolute Error")
        ax1.set_title("Absolute Error Distribution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")

        ax2 = fig.add_subplot(132)
        im2 = ax2.pcolormesh(X, Y, rel_error_grid, cmap="plasma")
        plt.colorbar(im2, ax=ax2, label="Relative Error")
        ax2.set_title("Relative Error Distribution")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_aspect("equal")

        ax3 = fig.add_subplot(133)
        log_error_grid = np.log10(error_grid + 1e-16)
        im3 = ax3.pcolormesh(X, Y, log_error_grid, cmap="viridis")
        plt.colorbar(im3, ax=ax3, label="log10(Absolute Error)")
        ax3.set_title("Log Error Distribution")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_aspect("equal")

        self._add_segment_boundaries_2d([ax1, ax2, ax3])
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _add_segment_boundaries_2d(self, axes_list: List) -> None:
        """为2DGraph添加段Boundary - Auxiliarymethod"""
        if not (hasattr(self.config, "x_min") and hasattr(self.config, "x_max")):
            return

        for index in range(getattr(self, "Ns", 0)):
            try:
                x_min = self.config.x_min[index]
                x_max = self.config.x_max[index]
            except (IndexError, AttributeError):
                continue
            for ax in axes_list:
                ax.plot([x_min[0], x_max[0]], [x_min[1], x_min[1]], "k--", alpha=0.4, linewidth=1)
                ax.plot([x_min[0], x_max[0]], [x_max[1], x_max[1]], "k--", alpha=0.4, linewidth=1)
                ax.plot([x_min[0], x_min[0]], [x_min[1], x_max[1]], "k--", alpha=0.4, linewidth=1)
                ax.plot([x_max[0], x_max[0]], [x_min[1], x_max[1]], "k--", alpha=0.4, linewidth=1)
