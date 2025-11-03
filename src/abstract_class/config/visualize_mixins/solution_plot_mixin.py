import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from scipy.interpolate import griddata


class SolutionPlotMixin:

    def plot_solution_1d(self, x: np.ndarray, u_true: np.ndarray, u_pred: np.ndarray,
                        save_path: Optional[str] = None, title: str = "1D Solution"):
        x = x.squeeze()
        u_true = u_true.squeeze()
        u_pred = u_pred.squeeze()

        sort_idx = np.argsort(x)
        x = x[sort_idx]
        u_true = u_true[sort_idx]
        u_pred = u_pred[sort_idx]
        abs_error = np.abs(u_pred - u_true)

        fig = self._create_figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

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

    def plot_solution_2d(self, x: np.ndarray, u_true: np.ndarray, u_pred: np.ndarray,
                        save_path: Optional[str] = None, title: str = "2D Solution",
                        n_grid: int = 100):
        u_true = u_true.squeeze()
        u_pred = u_pred.squeeze()

        x_coords = x[:, 0]
        y_coords = x[:, 1]
        abs_error = np.abs(u_pred - u_true)

        xi = np.linspace(np.min(x_coords), np.max(x_coords), n_grid)
        yi = np.linspace(np.min(y_coords), np.max(y_coords), n_grid)
        Xi, Yi = np.meshgrid(xi, yi)

        Ui_true = griddata((x_coords, y_coords), u_true, (Xi, Yi), method='cubic')
        Ui_pred = griddata((x_coords, y_coords), u_pred, (Xi, Yi), method='cubic')
        Ei = griddata((x_coords, y_coords), abs_error, (Xi, Yi), method='cubic')

        fig = self._create_figure(figsize=(15, 12))
        axes = [[fig.add_subplot(221), fig.add_subplot(222)],
                [fig.add_subplot(223), fig.add_subplot(224)]]

        c1 = axes[0][0].contourf(Xi, Yi, Ui_true, 50, cmap='viridis')
        axes[0][0].set_title('True Solution')
        axes[0][0].set_xlabel('x')
        axes[0][0].set_ylabel('y')
        fig.colorbar(c1, ax=axes[0][0])

        c2 = axes[0][1].contourf(Xi, Yi, Ui_pred, 50, cmap='viridis')
        axes[0][1].set_title('Predicted Solution')
        axes[0][1].set_xlabel('x')
        axes[0][1].set_ylabel('y')
        fig.colorbar(c2, ax=axes[0][1])

        c3 = axes[1][0].contourf(Xi, Yi, Ei, 50, cmap='hot')
        axes[1][0].set_title('Absolute Error')
        axes[1][0].set_xlabel('x')
        axes[1][0].set_ylabel('y')
        fig.colorbar(c3, ax=axes[1][0])

        mid_idx = n_grid // 2
        axes[1][1].semilogy(xi, Ei[mid_idx, :], 'b-', label='Error at y=mid')
        axes[1][1].semilogy(yi, Ei[:, mid_idx], 'r--', label='Error at x=mid')
        axes[1][1].set_title('Error Profiles (log scale)')
        axes[1][1].set_xlabel('Coordinate')
        axes[1][1].set_ylabel('Absolute Error')
        axes[1][1].legend()
        axes[1][1].grid(True)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def plot_solution_field(self, x: np.ndarray, y: np.ndarray, u: np.ndarray,
                           save_path: Optional[str] = None, title: str = "Solution Field",
                           cmap: str = 'viridis'):
        fig = self._create_figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        c = ax.contourf(x, y, u, 50, cmap=cmap)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        fig.colorbar(c, ax=ax)

        plt.tight_layout()
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _get_solution_colormap(self, var_name: str = None) -> str:
        cmap_dict = {
            'temperature': 'hot',
            'velocity': 'coolwarm',
            'pressure': 'viridis',
            'density': 'plasma',
        }
        return cmap_dict.get(var_name, 'viridis')
