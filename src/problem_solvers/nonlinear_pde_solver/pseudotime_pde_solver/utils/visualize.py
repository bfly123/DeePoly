import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os
from src.abstract_class.config.base_visualize import BaseVisualizer
from src.abstract_class.config.visualize_mixins import ConvergenceMixin, GridReconstructionMixin, ReferenceComparisonMixin
from src.visualization.dimensional_plotter import DimensionalVisualizer


class PseudoTimePDEVisualizer(BaseVisualizer, ConvergenceMixin, GridReconstructionMixin, ReferenceComparisonMixin):

    def __init__(self, config):
        super().__init__(config)
        # 使用维度可视化器进行智能维度判断
        self.dimensional_visualizer = DimensionalVisualizer(config)
        self.n_dim = self.dimensional_visualizer.n_dim

    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None):
        """使用维度可视化器自动选择绘图方法"""
        U_pred = self._to_numpy_cpu(prediction)

        if save_path:
            # 使用维度可视化器进行智能绘图
            viz_config = {
                'figsize': (10, 8),
                'colormap': getattr(self.config.visualization, 'colormap', 'RdBu_r'),
                'dpi': getattr(self.config.visualization, 'figure_dpi', 100)
            }
            self.dimensional_visualizer.plot_solution(U_pred, data, save_path, **viz_config)
        else:
            # 回退到原有方法
            fig = self._create_figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            x = data['x']

            if self.n_dim == 2:
                self._plot_2d(ax, x, U_pred)
            else:
                self._plot_1d(ax, x, U_pred)

            ax.set_title('Solution')
            plt.tight_layout()
            self._close_figure(fig)

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

        residual_history = conv_info['residual_history']
        iterations = list(range(len(residual_history)))
        tau_history = conv_info.get('tau_history', None)

        self.plot_residual_history(residual_history, iterations,
                                   os.path.join(save_dir, "convergence_history.png"),
                                   tau_history)

    def _save_solution_plot(self, output_name, U_pred, data, save_dir):
        """使用维度可视化器保存解图"""
        save_path = os.path.join(save_dir, f"{output_name}_solution.png")
        U_pred_cpu = self._to_numpy_cpu(U_pred)

        # 使用维度可视化器进行智能绘图
        viz_config = {
            'figsize': (10, 8),
            'colormap': getattr(self.config.visualization, 'colormap', 'RdBu_r'),
            'dpi': getattr(self.config.visualization, 'figure_dpi', 100),
            'plot_type': getattr(self.config.visualization, 'plot_type', 'contourf')
        }

        # 如果是2D问题，添加网格形状信息
        if self.n_dim == 2:
            try:
                x_unique = np.unique(data['x'][:, 0])
                y_unique = np.unique(data['x'][:, 1])
                viz_config['grid_shape'] = (len(y_unique), len(x_unique))
            except:
                viz_config['grid_shape'] = (100, 100)  # 默认形状

        self.dimensional_visualizer.plot_solution(U_pred_cpu, data, save_path, **viz_config)

    def _plot_2d(self, ax, x, U_pred):
        """2D绘图方法，使用正确的坐标轴标签"""
        x_unique = np.unique(x[:, 0])
        y_unique = np.unique(x[:, 1])
        X_mesh, Y_mesh = np.meshgrid(x_unique, y_unique, indexing='xy')
        U_grid = self._map_to_grid(x[:, 0], x[:, 1], U_pred.flatten(), x_unique, y_unique)
        im = ax.pcolormesh(X_mesh, Y_mesh, U_grid, cmap='RdBu_r', shading='gouraud')
        plt.colorbar(im, ax=ax, label='u')

        # 使用spatial_vars中的变量名作为坐标轴标签
        spatial_vars = getattr(self.config, 'spatial_vars', ['x', 't'])
        ax.set_xlabel(spatial_vars[0])
        ax.set_ylabel(spatial_vars[1] if len(spatial_vars) > 1 else 't')
        ax.set_aspect('equal')

    def _plot_1d(self, ax, x, U_pred):
        """1D绘图方法，使用正确的坐标轴标签"""
        x_flat = x[:, 0] if x.ndim > 1 else x
        u_flat = U_pred.flatten()
        sort_idx = np.argsort(x_flat)
        ax.plot(x_flat[sort_idx], u_flat[sort_idx], 'b-', linewidth=2)

        # 使用spatial_vars中的变量名作为坐标轴标签
        spatial_vars = getattr(self.config, 'spatial_vars', ['x'])
        ax.set_xlabel(spatial_vars[0])
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
