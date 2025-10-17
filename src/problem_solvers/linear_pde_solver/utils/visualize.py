import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from scipy.interpolate import griddata
from src.abstract_class.config.base_visualize import BaseVisualizer


class LinearPDEVisualizer(BaseVisualizer):

    def __init__(self, config=None):
        super().__init__(config)
        self.n_dim = config.n_dim if config else None

    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None, title: str = "PDE Solution"):
        if self.n_dim == 1:
            self._plot_1d_solution(data, prediction, save_path, title)
        elif self.n_dim == 2:
            self._plot_2d_solution(data, prediction, save_path, title)

    def _plot_1d_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None, title: str = "1D PDE Solution"):
        x = data["x"].squeeze()
        u_true = data["u"].squeeze()
        u_pred = prediction.squeeze()

        sort_idx = np.argsort(x)
        x = x[sort_idx]
        u_true = u_true[sort_idx]
        u_pred = u_pred[sort_idx]
        abs_error = np.abs(u_pred - u_true)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(x, u_true, 'b-', label='True Solution', linewidth=2)
        ax1.plot(x, u_pred, 'r--', label='Predicted Solution', linewidth=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True)

        ax2.semilogy(x, abs_error, 'k-', linewidth=1.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Absolute Error (log scale)')
        ax2.set_title('Absolute Error')
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _plot_2d_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None, title: str = "2D PDE Solution"):
        x = data["x"]
        u_true = data["u"].squeeze()
        u_pred = prediction.squeeze()

        x_coords = x[:, 0]
        y_coords = x[:, 1]
        abs_error = np.abs(u_pred - u_true)

        n_grid = 100
        xi = np.linspace(np.min(x_coords), np.max(x_coords), n_grid)
        yi = np.linspace(np.min(y_coords), np.max(y_coords), n_grid)
        Xi, Yi = np.meshgrid(xi, yi)

        Ui_true = griddata((x_coords, y_coords), u_true, (Xi, Yi), method='cubic')
        Ui_pred = griddata((x_coords, y_coords), u_pred, (Xi, Yi), method='cubic')
        Ei = griddata((x_coords, y_coords), abs_error, (Xi, Yi), method='cubic')

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        c1 = axes[0, 0].contourf(Xi, Yi, Ui_true, 50, cmap='viridis')
        axes[0, 0].set_title('True Solution')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        fig.colorbar(c1, ax=axes[0, 0])

        c2 = axes[0, 1].contourf(Xi, Yi, Ui_pred, 50, cmap='viridis')
        axes[0, 1].set_title('Predicted Solution')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        fig.colorbar(c2, ax=axes[0, 1])

        c3 = axes[1, 0].contourf(Xi, Yi, Ei, 50, cmap='hot')
        axes[1, 0].set_title('Absolute Error')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        fig.colorbar(c3, ax=axes[1, 0])

        mid_idx = n_grid // 2
        axes[1, 1].semilogy(xi, Ei[mid_idx, :], 'b-', label='Error at y=mid')
        axes[1, 1].semilogy(yi, Ei[:, mid_idx], 'r--', label='Error at x=mid')
        axes[1, 1].set_title('Error Profiles (log scale)')
        axes[1, 1].set_xlabel('Coordinate')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def plot_convergence(self, errors: list, save_path: Optional[str] = None, title: str = "Convergence History"):
        fig = self._create_figure(figsize=(10, 6))
        plt.semilogy(range(1, len(errors) + 1), errors, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Error (log scale)')
        plt.title(title)
        plt.grid(True)
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)
