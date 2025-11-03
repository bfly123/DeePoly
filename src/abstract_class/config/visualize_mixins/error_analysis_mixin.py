import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from scipy.interpolate import griddata


class ErrorAnalysisMixin:

    def plot_error_field_2d(self, x: np.ndarray, u_pred: np.ndarray, u_true: np.ndarray,
                           save_path: Optional[str] = None, title: str = "Error Field Analysis",
                           n_grid: int = 100):
        abs_error = np.abs(u_pred - u_true).squeeze()
        rel_error = (abs_error / (np.abs(u_true.squeeze()) + 1e-10))

        x_coords = x[:, 0]
        y_coords = x[:, 1]

        xi = np.linspace(np.min(x_coords), np.max(x_coords), n_grid)
        yi = np.linspace(np.min(y_coords), np.max(y_coords), n_grid)
        Xi, Yi = np.meshgrid(xi, yi)

        Ei_abs = griddata((x_coords, y_coords), abs_error, (Xi, Yi), method='cubic')
        Ei_rel = griddata((x_coords, y_coords), rel_error, (Xi, Yi), method='cubic')

        fig = self._create_figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        c1 = ax1.contourf(Xi, Yi, Ei_abs, 50, cmap='hot')
        ax1.set_title('Absolute Error Field')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        fig.colorbar(c1, ax=ax1)

        c2 = ax2.contourf(Xi, Yi, Ei_rel, 50, cmap='hot')
        ax2.set_title('Relative Error Field')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        fig.colorbar(c2, ax=ax2)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def analyze_residual_components(self, residuals: Dict[str, np.ndarray],
                                   save_path: Optional[str] = None,
                                   title: str = "Residual Component Analysis"):
        fig = self._create_figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        for name, res in residuals.items():
            ax1.semilogy(res, label=name, linewidth=2)

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Residual (log scale)')
        ax1.set_title('Individual Residual Components')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        total_residual = np.sqrt(sum(r**2 for r in residuals.values()))
        ax2.semilogy(total_residual, 'k-', linewidth=2, label='Total')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Total Residual (log scale)')
        ax2.set_title('Combined Residual')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def compute_spatial_error_stats(self, x: np.ndarray, errors: np.ndarray) -> Dict:
        if x.shape[1] == 1:
            x_coord = x.squeeze()
            sort_idx = np.argsort(x_coord)
            x_sorted = x_coord[sort_idx]
            errors_sorted = errors.squeeze()[sort_idx]

            n_bins = 20
            bin_edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
            bin_stats = []

            for i in range(n_bins):
                mask = (x_sorted >= bin_edges[i]) & (x_sorted < bin_edges[i+1])
                if mask.sum() > 0:
                    bin_errors = errors_sorted[mask]
                    bin_stats.append({
                        'x_center': (bin_edges[i] + bin_edges[i+1]) / 2,
                        'mean': np.mean(bin_errors),
                        'std': np.std(bin_errors),
                        'max': np.max(bin_errors),
                    })

            return {'dimension': 1, 'bins': bin_stats}

        elif x.shape[1] == 2:
            return {
                'dimension': 2,
                'global_mean': np.mean(errors),
                'global_std': np.std(errors),
                'global_max': np.max(errors),
            }

    def plot_error_statistics(self, spatial_stats: Dict, save_path: Optional[str] = None,
                             title: str = "Spatial Error Statistics"):
        if spatial_stats['dimension'] == 1:
            x_centers = [b['x_center'] for b in spatial_stats['bins']]
            means = [b['mean'] for b in spatial_stats['bins']]
            stds = [b['std'] for b in spatial_stats['bins']]
            maxs = [b['max'] for b in spatial_stats['bins']]

            fig = self._create_figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            ax.semilogy(x_centers, means, 'b-', linewidth=2, label='Mean Error')
            ax.fill_between(x_centers,
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.3, label='±1 Std Dev')
            ax.semilogy(x_centers, maxs, 'r--', linewidth=2, label='Max Error')

            ax.set_xlabel('Spatial Location')
            ax.set_ylabel('Error (log scale)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_path:
                self._save_figure(fig, save_path)
            self._close_figure(fig)
