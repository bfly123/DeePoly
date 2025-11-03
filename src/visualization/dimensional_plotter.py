"""
基于维度的可视化绘图模块
根据config.spatial_vars自动判断问题维度并选择合适的可视化方法
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List
import os


class DimensionalVisualizer:
    """
    智能维度可视化器
    根据spatial_vars自动判断维度并选择绘图方式
    """

    def __init__(self, config):
        self.config = config
        self.spatial_vars = getattr(config, 'spatial_vars', ['x'])
        self.n_dim = len(self.spatial_vars)

    def plot_solution(self, U: np.ndarray, data: Dict[str, Any],
                     save_path: str, **kwargs) -> None:
        """
        根据维度自动选择绘图方法

        Args:
            U: 解场 (n_points, n_eqs)
            data: 数据字典，包含坐标等信息
            save_path: 保存路径
            **kwargs: 其他绘图参数
        """
        # 如果save_path不是绝对路径，尝试使用config中的results_dir
        if not os.path.isabs(save_path) and hasattr(self.config, 'results_dir'):
            save_path = os.path.join(self.config.results_dir, os.path.basename(save_path))

        if self.n_dim == 1:
            self._plot_1d_solution(U, data, save_path, **kwargs)
        elif self.n_dim == 2:
            self._plot_2d_solution(U, data, save_path, **kwargs)
        else:
            raise ValueError(f"Unsupported dimension: {self.n_dim}")

    def _plot_1d_solution(self, U: np.ndarray, data: Dict[str, Any],
                         save_path: str, **kwargs) -> None:
        """绘制1D解"""
        x = data['x'].flatten()
        n_eqs = U.shape[1] if len(U.shape) > 1 else 1

        plt.figure(figsize=kwargs.get('figsize', (10, 6)))

        if n_eqs == 1:
            y = U.flatten() if len(U.shape) > 1 else U
            plt.plot(x, y, linewidth=2)
            plt.xlabel(self.spatial_vars[0])
            plt.ylabel('u')
            plt.title('1D Solution')
        else:
            for i in range(n_eqs):
                y = U[:, i].flatten()
                plt.plot(x, y, label=f'u_{i}', linewidth=2)
            plt.xlabel(self.spatial_vars[0])
            plt.ylabel('u')
            plt.title('1D Solution (Multiple Variables)')
            plt.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=kwargs.get('dpi', 100), bbox_inches='tight')
        plt.close()

    def _plot_2d_solution(self, U: np.ndarray, data: Dict[str, Any],
                         save_path: str, **kwargs) -> None:
        """绘制2D解"""
        x = data['x']
        n_eqs = U.shape[1] if len(U.shape) > 1 else 1

        if x.shape[1] != 2:
            raise ValueError("2D plotting requires exactly 2 spatial coordinates")

        # 重构网格
        x_coords = x[:, 0].reshape(kwargs.get('grid_shape', (100, 100)))
        y_coords = x[:, 1].reshape(kwargs.get('grid_shape', (100, 100)))

        for eq_idx in range(n_eqs):
            if n_eqs > 1:
                save_path_eq = save_path.replace('.png', f'_eq{eq_idx}.png')
            else:
                save_path_eq = save_path

            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))

            # 提取解场 - 处理1D和多变量情况
            if n_eqs == 1:
                u_field = U.flatten().reshape(kwargs.get('grid_shape', (100, 100)))
            else:
                u_field = U[:, eq_idx].reshape(kwargs.get('grid_shape', (100, 100)))

            # 绘制等高线图或填色图
            plot_type = kwargs.get('plot_type', 'contourf')
            levels = kwargs.get('levels', 20)
            colormap = kwargs.get('colormap', 'RdBu_r')

            if plot_type == 'contourf':
                contour = ax.contourf(x_coords, y_coords, u_field,
                                     levels=levels, cmap=colormap)
                plt.colorbar(contour, ax=ax, label='u')
            elif plot_type == 'contour':
                contour = ax.contour(x_coords, y_coords, u_field,
                                   levels=levels, cmap=colormap)
                plt.colorbar(contour, ax=ax, label='u')
            else:  # pcolormesh
                im = ax.pcolormesh(x_coords, y_coords, u_field,
                                  cmap=colormap, shading='auto')
                plt.colorbar(im, ax=ax, label='u')

            ax.set_xlabel(self.spatial_vars[0])
            ax.set_ylabel(self.spatial_vars[1])
            title = f'2D Solution' if n_eqs == 1 else f'2D Solution (Variable {eq_idx})'
            ax.set_title(title)

            plt.tight_layout()
            if os.path.dirname(save_path_eq):
                    os.makedirs(os.path.dirname(save_path_eq), exist_ok=True)
            plt.savefig(save_path_eq, dpi=kwargs.get('dpi', 100), bbox_inches='tight')
            plt.close()

    def plot_comparison_with_reference(self, U_numerical: np.ndarray, data: Dict[str, Any],
                                    reference_solution: Optional[Dict[str, Any]],
                                    save_path: str, **kwargs) -> None:
        """
        与参考解对比

        Args:
            U_numerical: 数值解
            data: 数据字典
            reference_solution: 参考解字典，包含'x_ref', 'u_ref'等
            save_path: 保存路径
            **kwargs: 其他参数
        """
        if reference_solution is None:
            self.plot_solution(U_numerical, data, save_path, **kwargs)
            return

        if self.n_dim == 1:
            self._plot_1d_comparison(U_numerical, data, reference_solution, save_path, **kwargs)
        elif self.n_dim == 2:
            self._plot_2d_comparison(U_numerical, data, reference_solution, save_path, **kwargs)
        else:
            raise ValueError(f"Unsupported dimension: {self.n_dim}")

    def _plot_1d_comparison(self, U_numerical: np.ndarray, data: Dict[str, Any],
                          reference_solution: Dict[str, Any], save_path: str, **kwargs) -> None:
        """1D解对比"""
        x_num = data['x'].flatten()
        x_ref = reference_solution['x_ref'].flatten()
        u_num = U_numerical.flatten() if len(U_numerical.shape) > 1 else U_numerical
        u_ref = reference_solution['u_ref'].flatten()

        plt.figure(figsize=kwargs.get('figsize', (12, 6)))

        plt.subplot(1, 2, 1)
        plt.plot(x_num, u_num, 'b-', label='Numerical', linewidth=2)
        plt.plot(x_ref, u_ref, 'r--', label='Reference', linewidth=2)
        plt.xlabel(self.spatial_vars[0])
        plt.ylabel('u')
        plt.title('Solution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # 插值到相同网格进行误差计算
        from scipy.interpolate import interp1d
        if len(x_ref) > len(x_num):
            f_interp = interp1d(x_ref, u_ref, kind='linear', fill_value='extrapolate')
            u_ref_interp = f_interp(x_num)
            error = np.abs(u_num - u_ref_interp)
        else:
            f_interp = interp1d(x_num, u_num, kind='linear', fill_value='extrapolate')
            u_num_interp = f_interp(x_ref)
            error = np.abs(u_num_interp - u_ref)

        plt.plot(x_num if len(x_ref) > len(x_num) else x_ref, error, 'g-', linewidth=2)
        plt.xlabel(self.spatial_vars[0])
        plt.ylabel('Absolute Error')
        plt.title('Error Analysis')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=kwargs.get('dpi', 100), bbox_inches='tight')
        plt.close()

    def _plot_2d_comparison(self, U_numerical: np.ndarray, data: Dict[str, Any],
                          reference_solution: Dict[str, Any], save_path: str, **kwargs) -> None:
        """2D解对比"""
        # 检查维度并处理可能的降维
        if 'x_ref' in reference_solution and len(reference_solution['x_ref'].shape) == 2:
            if reference_solution['x_ref'].shape[1] == 3:  # 2D+时间，需要时间切片
                time_slice = kwargs.get('time_slice', 0.5)  # 默认取中间时刻
                ref_solution_sliced = self._slice_2d_to_2d(reference_solution, time_slice)
            else:
                ref_solution_sliced = reference_solution
        else:
            ref_solution_sliced = reference_solution

        # 绘制数值解
        grid_shape = kwargs.get('grid_shape', (100, 100))
        self._plot_2d_solution(U_numerical, data,
                             save_path.replace('.png', '_numerical.png'),
                             **kwargs)

        # 绘制参考解
        if 'u_ref' in ref_solution_sliced:
            x_ref = ref_solution_sliced['x_ref']
            u_ref = ref_solution_sliced['u_ref']

            ref_data = {'x': x_ref}
            self._plot_2d_solution(u_ref, ref_data,
                                 save_path.replace('.png', '_reference.png'),
                                 **kwargs)

            # 计算并绘制误差
            if self._can_compare_2d_grids(data, ref_solution_sliced, grid_shape):
                error_data = self._compute_2d_error(U_numerical, data, u_ref,
                                                  ref_solution_sliced, grid_shape)
                self._plot_2d_solution(error_data['error'], error_data,
                                     save_path.replace('.png', '_error.png'),
                                     **{**kwargs, 'colormap': 'viridis'})

    def _slice_2d_to_2d(self, ref_solution: Dict[str, Any], time_value: float) -> Dict[str, Any]:
        """从2D+时间参考解中提取时间切片"""
        x_ref = ref_solution['x_ref']
        u_ref = ref_solution['u_ref']

        # 假设最后一维是时间
        if len(x_ref.shape) == 3 and x_ref.shape[2] > 1:
            # 找到最接近time_value的时间索引
            time_coords = x_ref[0, 0, :] if len(u_ref.shape) == 3 else x_ref[0, :, -1]
            time_idx = np.argmin(np.abs(time_coords - time_value))

            # 提取切片
            if len(u_ref.shape) == 3:
                u_slice = u_ref[:, :, time_idx]
            else:
                u_slice = u_ref[:, time_idx]

            return {
                'x_ref': x_ref[:, :, :time_idx+1].reshape(-1, 2) if time_idx > 0 else x_ref[:, :, 0].reshape(-1, 2),
                'u_ref': u_slice.flatten() if len(u_slice.shape) > 1 else u_slice
            }

        return ref_solution

    def _can_compare_2d_grids(self, data_num: Dict[str, Any],
                            ref_solution: Dict[str, Any],
                            grid_shape: Tuple[int, int]) -> bool:
        """检查是否可以比较2D网格"""
        try:
            x_num = data_num['x']
            x_ref = ref_solution['x_ref']

            # 简化检查：看网格点数量是否匹配
            return x_num.shape[0] == x_ref.shape[0]
        except:
            return False

    def _compute_2d_error(self, U_numerical: np.ndarray, data_num: Dict[str, Any],
                        u_ref: np.ndarray, ref_solution: Dict[str, Any],
                        grid_shape: Tuple[int, int]) -> Dict[str, Any]:
        """计算2D误差场"""
        u_num_reshaped = U_numerical.reshape(grid_shape)
        u_ref_reshaped = u_ref.reshape(grid_shape)
        error = np.abs(u_num_reshaped - u_ref_reshaped)

        return {
            'x': data_num['x'],
            'error': error.flatten()
        }

    def get_dimension_info(self) -> Dict[str, Any]:
        """获取维度信息"""
        return {
            'n_dim': self.n_dim,
            'spatial_vars': self.spatial_vars,
            'plot_type': '1D line plot' if self.n_dim == 1 else '2D contour plot'
        }