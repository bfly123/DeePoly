import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Callable
from scipy.interpolate import RegularGridInterpolator


class ReferenceComparisonMixin:

    def _resolve_reference_values(self, x: np.ndarray, reference_func):
        if isinstance(reference_func, dict):
            if 'u_ref' not in reference_func:
                print("Warning: Reference solution dict missing 'u_ref' key, skipping comparison")
                return None
            u_ref = np.asarray(reference_func['u_ref'])
            y_ref = reference_func.get('y_ref')
            x_ref = reference_func.get('x_ref')

            if (
                y_ref is not None
                and x_ref is not None
                and x.ndim == 2
                and x.shape[1] >= 2
            ):
                x_ref = np.asarray(x_ref).reshape(-1)
                y_ref = np.asarray(y_ref).reshape(-1)
                if u_ref.ndim == 1 and u_ref.size == x_ref.size * y_ref.size:
                    u_ref = u_ref.reshape(len(y_ref), len(x_ref))
                elif u_ref.shape == (len(x_ref), len(y_ref)):
                    u_ref = u_ref.T
                elif u_ref.shape != (len(y_ref), len(x_ref)):
                    raise ValueError(
                        f"Unsupported 2D reference shape {u_ref.shape}; "
                        f"expected {(len(y_ref), len(x_ref))}"
                    )

                interpolator = RegularGridInterpolator(
                    (y_ref, x_ref),
                    u_ref,
                    bounds_error=False,
                    fill_value=None,
                )
                query = np.column_stack([x[:, 1], x[:, 0]])
                resolved = interpolator(query)
                return resolved.reshape(-1, 1)

            if u_ref.ndim == 1:
                return u_ref.reshape(-1, 1)
            return u_ref

        u_ref = reference_func(x)
        u_ref = np.asarray(u_ref)
        if u_ref.ndim == 1:
            return u_ref.reshape(-1, 1)
        return u_ref

    def plot_reference_comparison(self, x: np.ndarray, u_pred: np.ndarray,
                                  reference_func, save_path: Optional[str] = None,
                                  title: str = "Reference Comparison"):
        u_ref = self._resolve_reference_values(x, reference_func)
        if u_ref is None:
            return
        u_pred = np.asarray(u_pred)
        if u_pred.ndim == 1:
            u_pred = u_pred.reshape(-1, 1)
        abs_error = np.abs(u_pred - u_ref)
        rel_error = abs_error / (np.abs(u_ref) + 1e-10)

        fig = self._create_figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        if x.shape[1] == 1:
            x_plot = x.squeeze()
            sort_idx = np.argsort(x_plot)
            x_plot = x_plot[sort_idx]
            u_pred_plot = u_pred.squeeze()[sort_idx]
            u_ref_plot = u_ref.squeeze()[sort_idx]

            ax1.plot(x_plot, u_ref_plot, 'b-', label='Reference', linewidth=2)
            ax1.plot(x_plot, u_pred_plot, 'r--', label='Prediction', linewidth=2)
            ax1.set_xlabel('x')
            ax1.set_ylabel('u')
            ax1.legend()
            ax1.grid(True)

            ax2.semilogy(x_plot, abs_error.squeeze()[sort_idx], 'k-', linewidth=1.5)
            ax2.set_xlabel('x')
            ax2.set_ylabel('Absolute Error')
            ax2.grid(True)

            ax3.semilogy(x_plot, rel_error.squeeze()[sort_idx], 'g-', linewidth=1.5)
            ax3.set_xlabel('x')
            ax3.set_ylabel('Relative Error')
            ax3.grid(True)
        elif x.shape[1] == 2:
            pred_plot = u_pred[:, 0]
            ref_plot = u_ref[:, 0]
            err_plot = abs_error[:, 0]

            scat1 = ax1.scatter(x[:, 0], x[:, 1], c=ref_plot, s=10, cmap='viridis')
            ax1.set_title('Reference')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.grid(True, alpha=0.3)
            fig.colorbar(scat1, ax=ax1)

            scat2 = ax2.scatter(x[:, 0], x[:, 1], c=pred_plot, s=10, cmap='viridis')
            ax2.set_title('Prediction')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.grid(True, alpha=0.3)
            fig.colorbar(scat2, ax=ax2)

            scat3 = ax3.scatter(x[:, 0], x[:, 1], c=err_plot, s=10, cmap='magma')
            ax3.set_title('Absolute Error')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.grid(True, alpha=0.3)
            fig.colorbar(scat3, ax=ax3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)

    def compute_error_metrics(self, u_pred: np.ndarray, u_true: np.ndarray) -> Dict[str, float]:
        abs_error = np.abs(u_pred - u_true)
        rel_error = abs_error / (np.abs(u_true) + 1e-10)

        metrics = {
            'l1_error': np.mean(abs_error),
            'l2_error': np.sqrt(np.mean(abs_error**2)),
            'linf_error': np.max(abs_error),
            'mean_rel_error': np.mean(rel_error),
            'max_rel_error': np.max(rel_error),
        }
        return metrics

    def plot_error_distribution(self, u_pred: np.ndarray, u_true: np.ndarray,
                               save_path: Optional[str] = None, title: str = "Error Distribution"):
        abs_error = np.abs(u_pred - u_true).flatten()
        rel_error = (abs_error / (np.abs(u_true.flatten()) + 1e-10))

        fig = self._create_figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.hist(abs_error, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Absolute Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Absolute Error Distribution')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        ax2.hist(rel_error, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Relative Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Relative Error Distribution')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)
