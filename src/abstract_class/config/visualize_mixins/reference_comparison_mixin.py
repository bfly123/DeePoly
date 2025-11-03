import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Callable


class ReferenceComparisonMixin:

    def plot_reference_comparison(self, x: np.ndarray, u_pred: np.ndarray,
                                  reference_func, save_path: Optional[str] = None,
                                  title: str = "Reference Comparison"):
        # 处理参考解可能是字典或函数的情况
        if isinstance(reference_func, dict):
            if 'u_ref' in reference_func:
                u_ref = reference_func['u_ref']
            else:
                # 如果是字典但没有u_ref，跳过比较
                print("Warning: Reference solution dict missing 'u_ref' key, skipping comparison")
                return
        else:
            u_ref = reference_func(x)
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
