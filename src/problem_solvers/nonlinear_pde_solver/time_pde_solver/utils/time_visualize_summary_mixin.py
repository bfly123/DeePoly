from collections import deque
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class TimePDESummaryMixin:
    def plot_loss_history(self, loss_history: List[float], save_path: Optional[str] = None) -> None:
        """绘制TrainingLossHistory"""
        fig = self._create_figure(figsize=(10, 6))
        iterations = range(1, len(loss_history) + 1)
        ax = fig.add_subplot(111)
        ax.semilogy(iterations, loss_history, "b-", linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Iterate步数", fontsize=12)
        ax.set_ylabel("Lossvalue (对数尺Degree)", fontsize=12)
        ax.set_title("TrainingLossHistory", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if loss_history:
            final_loss = loss_history[-1]
            min_loss = min(loss_history)
            ax.text(
                0.02,
                0.98,
                f"FinalLoss: {final_loss:.2e}\n最小Loss: {min_loss:.2e}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def plot_time_evolution_summary(
        self,
        time_history: List[float],
        solution_history: List[np.ndarray],
        save_path: Optional[str] = None,
    ) -> None:
        """绘制Time演化SummaryGraph"""
        if not time_history or not solution_history:
            print("Warning: 没有Time演化Data可绘制")
            return
        handler = (
            self._plot_time_evolution_summary_1d
            if self.n_dim == 1
            else self._plot_time_evolution_summary_2d
        )
        handler(time_history, solution_history, save_path)

    def _normalize_summary_histories(self, time_history, solution_history):
        if len(time_history) != len(solution_history):
            print(
                f"Warning: TimeHistoryLength({len(time_history)})与"
                f"SolutionHistoryLength({len(solution_history)})不匹配"
            )
            min_len = min(len(time_history), len(solution_history))
            return time_history[:min_len], solution_history[:min_len]
        return time_history, solution_history

    def _snapshot_indices(self, time_history: List[float]) -> np.ndarray:
        n_snapshots = min(6, len(time_history))
        if len(time_history) == 1:
            return np.array([0])
        return np.linspace(0, len(time_history) - 1, n_snapshots, dtype=int)

    def _plot_time_evolution_summary_1d(
        self,
        time_history: List[float],
        solution_history: List[np.ndarray],
        save_path: Optional[str] = None,
    ) -> None:
        """1DTime演化SummaryGraph"""
        fig = self._create_figure(figsize=(15, 10))
        time_history, solution_history = self._normalize_summary_histories(time_history, solution_history)
        if len(time_history) == 0:
            print("Warning: 没有Time演化Data")
            return
        for plot_index, history_index in enumerate(self._snapshot_indices(time_history), start=1):
            ax = fig.add_subplot(2, 3, plot_index)
            u = solution_history[history_index]
            t = time_history[history_index]
            if hasattr(self.config, "x_domain") and hasattr(self.config, "points_domain_test"):
                x = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], len(u))
                ax.plot(x, u.flatten(), "b-", linewidth=2)
            else:
                ax.plot(u.flatten(), "b-", linewidth=2)
            ax.set_title(f"T = {t:.4f}", fontsize=12)
            ax.grid(True, alpha=0.3)
            if plot_index >= 4:
                ax.set_xlabel("x", fontsize=10)
            if plot_index in {1, 4}:
                ax.set_ylabel("u", fontsize=10)
        plt.suptitle("Time演化快照", fontsize=16, fontweight="bold")
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _plot_time_evolution_summary_2d(
        self,
        time_history: List[float],
        solution_history: List[np.ndarray],
        save_path: Optional[str] = None,
    ) -> None:
        """2DTime演化SummaryGraph"""
        fig = self._create_figure(figsize=(15, 10))
        time_history, solution_history = self._normalize_summary_histories(time_history, solution_history)
        if len(time_history) == 0:
            print("Warning: 没有Time演化Data")
            return
        for plot_index, history_index in enumerate(self._snapshot_indices(time_history), start=1):
            ax = fig.add_subplot(2, 3, plot_index)
            u = solution_history[history_index]
            t = time_history[history_index]
            if u.ndim == 1:
                side_length = int(np.sqrt(len(u)))
                if side_length * side_length == len(u):
                    u_grid = u.reshape(side_length, side_length)
                    im = ax.imshow(u_grid, cmap="viridis", aspect="equal")
                    plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"T = {t:.4f}", fontsize=12)
            ax.set_aspect("equal")
        plt.suptitle("Time演化快照", fontsize=16, fontweight="bold")
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def set_visualization_interval(self, interval: int = 1) -> None:
        """SetupVisualizationBetween隔"""
        self._visualization_interval = max(1, interval)
        print(f"VisualizationBetween隔Setup为: 每 {self._visualization_interval} 次IterateUpdateOnce")

    def reset_animation_data(self):
        """ResetanimationData"""
        self.time_history_viz = deque(maxlen=self._max_frames)
        self.solution_history_viz = deque(maxlen=self._max_frames)
        self.coeffs_history_viz = deque(maxlen=self._max_frames)
        self._frames = []
        print("animationData已Reset")
