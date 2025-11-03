import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
import os


class ConvergenceMixin:

    def plot_residual_history(self, residuals: List[float], iterations: List[int],
                             save_path: Optional[str] = None, tau_history: Optional[List[float]] = None):
        fig = self._create_figure(figsize=(14, 5) if tau_history else (10, 6))

        if tau_history:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.semilogy(iterations, residuals, 'b-', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Residual (log)')
            ax1.set_title('Residual History')
            ax1.grid(True, alpha=0.3)

            ax2.semilogy(tau_history, residuals, 'r-', linewidth=2)
            ax2.set_xlabel('Pseudo-time τ')
            ax2.set_ylabel('Residual (log)')
            ax2.set_title('Residual vs Pseudo-time')
            ax2.grid(True, alpha=0.3)
        else:
            ax = fig.add_subplot(111)
            ax.semilogy(iterations, residuals, 'r-', linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Residual (log scale)')
            ax.set_title('Convergence History')
            ax.grid(True, alpha=0.3)

            if residuals:
                stats = self._format_convergence_stats(residuals)
                ax.text(0.02, 0.98, stats, transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def plot_convergence_rate(self, residuals: List[float], save_path: Optional[str] = None):
        residuals_arr = np.array(residuals)
        rate = -np.diff(np.log(residuals_arr + 1e-16))

        fig = self._create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(rate, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Convergence Rate')
        ax.set_title('Local Convergence Rate')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def save_convergence_history(self, conv_info: Dict, save_dir: str):
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

        if "solution_norm_history" in conv_info:
            axes[1, 0].plot(conv_info["solution_norm_history"])
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("||U||")
            axes[1, 0].set_title("Solution Norm Evolution")
            axes[1, 0].grid(True, alpha=0.3)

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

    def _format_convergence_stats(self, residuals: List[float]) -> str:
        final = residuals[-1]
        initial = residuals[0]
        return f'Initial: {initial:.2e}\nFinal: {final:.2e}\nReduction: {initial/final:.2e}x'
