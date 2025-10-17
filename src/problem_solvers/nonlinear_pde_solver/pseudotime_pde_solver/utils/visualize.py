import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os
from src.abstract_class.config.base_visualize import BaseVisualizer


class PseudoTimePDEVisualizer(BaseVisualizer):

    def __init__(self, config):
        super().__init__(config)
        self.n_dim = len(config.x_domain)

    def render(self, results: Dict, data_train: Dict, data_test: Dict,
               reference_solution: Optional[Dict] = None):
        results_dir = self.config.results_dir
        outputs = self.config.visualization.outputs
        predictions = results['predictions']

        self._save_convergence_plots(results['convergence_info'], results_dir)

        for output_name in outputs:
            pred = predictions[output_name]
            self._save_solution_plot(output_name, pred, data_test, results_dir)

            if reference_solution and self._should_plot_reference():
                self.plot_reference_comparison(pred, data_test, reference_solution,
                                               results_dir, prefix=output_name)

        self._save_data(predictions, results, data_test, results_dir)
        print(f"\n✓ All visualizations saved to: {results_dir}")

    def _save_convergence_plots(self, conv_info, save_dir):
        if 'residual_history' not in conv_info or not conv_info['residual_history']:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        residual_history = conv_info['residual_history']
        iterations = range(len(residual_history))

        axes[0].semilogy(iterations, residual_history, 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Residual (log)')
        axes[0].set_title('Residual History')
        axes[0].grid(True, alpha=0.3)

        if 'tau_history' in conv_info:
            axes[1].semilogy(conv_info['tau_history'], residual_history, 'r-', linewidth=2)
            axes[1].set_xlabel('Pseudo-time τ')
            axes[1].set_ylabel('Residual (log)')
            axes[1].set_title('Residual vs Pseudo-time')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, os.path.join(save_dir, "convergence_history.png"))

    def _save_solution_plot(self, output_name, U_pred, data, save_dir):
        fig = self._create_figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        x = data['x']

        if self.n_dim == 2:
            self._plot_2d(ax, x, U_pred)
        else:
            self._plot_1d(ax, x, U_pred)

        ax.set_title(f'{output_name.capitalize()} Solution')
        plt.tight_layout()
        self._save_figure(fig, os.path.join(save_dir, f"{output_name}_solution.png"))

    def _plot_2d(self, ax, x, U_pred):
        x_unique = np.unique(x[:, 0])
        y_unique = np.unique(x[:, 1])
        X_mesh, Y_mesh = np.meshgrid(x_unique, y_unique, indexing='xy')
        U_grid = self._map_to_grid(x[:, 0], x[:, 1], U_pred.flatten(), x_unique, y_unique)
        im = ax.pcolormesh(X_mesh, Y_mesh, U_grid, cmap='RdBu_r', shading='gouraud')
        plt.colorbar(im, ax=ax, label='u')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_aspect('equal')

    def _plot_1d(self, ax, x, U_pred):
        x_flat = x[:, 0] if x.ndim > 1 else x
        u_flat = U_pred.flatten()
        sort_idx = np.argsort(x_flat)
        ax.plot(x_flat[sort_idx], u_flat[sort_idx], 'b-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.grid(True, alpha=0.3)

    def _save_data(self, predictions, results, data_test, save_dir):
        save_data = {
            'x_test': data_test['x'],
            'coeffs': self._to_numpy_cpu(results['coefficients']),
            'converged': results['convergence_info']['converged']
        }
        for output_name, U_pred in predictions.items():
            save_data[f'{output_name}_test'] = self._to_numpy_cpu(U_pred)

        np.savez(os.path.join(save_dir, "solution_data.npz"), **save_data)
        print(f"  Data saved: {os.path.join(save_dir, 'solution_data.npz')}")

    def _should_plot_reference(self):
        return self.config.visualization.reference_policy in ['auto', 'force']

    def update_animation_data(self, iteration, residual, U, data, skip):
        pass
