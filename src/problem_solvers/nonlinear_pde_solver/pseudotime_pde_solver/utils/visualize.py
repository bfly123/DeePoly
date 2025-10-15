import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from PIL import Image
import io
import os
import torch
from scipy.interpolate import griddata, interp1d
from matplotlib.colors import LogNorm
from src.abstract_class.config.base_visualize import BaseVisualizer


class PseudoTimePDEVisualizer(BaseVisualizer):
    """Visualizer for general PDE problems with pseudo-time stepping

    Extends BaseVisualizer with specific functionality for:
    - Pseudo-time stepping iteration visualization
    - Residual convergence monitoring
    - Dual-panel displays (solution + residual)
    """

    def __init__(self, config):
        super().__init__(config)
        self.n_dim = len(config.spatial_vars) if hasattr(config, 'spatial_vars') else len(config.x_domain)

        # 历史数据
        self.iteration_history = []
        self.residual_history = []
        self.solution_history = []

        # 用于存储动画帧
        self._frames = []
        self._save_gif = False

        # 用于实时更新的图形对象
        self._scatter_solution = None
        self._line_residual = None
        self._colorbar = None  # 用于2D可视化的颜色条

        # 实时可视化窗口（仅在需要时创建）
        self.fig = None
        self.ax_solution = None
        self.ax_residual = None
        self._realtime_mode_active = False

    def reset_animation_data(self):
        """重置动画数据"""
        self.iteration_history = []
        self.residual_history = []
        self.solution_history = []
        self._frames = []
        print("动画数据已重置")

    def _to_numpy_cpu(self, data):
        """Convert data to CPU numpy array (handle torch tensors and GPU arrays)

        Args:
            data: Input data (numpy array, torch tensor, or GPU array)

        Returns:
            numpy array on CPU
        """
        if data is None:
            return None

        # Handle torch tensors
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()

        # Handle numpy arrays (might be on GPU via cupy)
        if isinstance(data, np.ndarray):
            # Already numpy array, check if it's a view of GPU memory
            return np.asarray(data)

        # Handle other array-like objects
        try:
            return np.asarray(data)
        except:
            return data

    def update_animation_data(self, iteration: int, residual: float, U: np.ndarray,
                            data_train: Dict, animation_skip: int = 1) -> None:
        """更新动画数据"""
        if iteration % animation_skip == 0:
            # Ensure U is on CPU before storing
            U_cpu = self._to_numpy_cpu(U)
            self.iteration_history.append(iteration)
            self.residual_history.append(residual)
            self.solution_history.append(U_cpu.copy())

            # 实时可视化
            if getattr(self.config, 'realtime_visualization', False):
                try:
                    x_coords = data_train.get('x', data_train.get('x_segments', []))
                    if isinstance(x_coords, list):
                        if x_coords:
                            x_plot = np.vstack(x_coords)
                        else:
                            print(f"  Warning: Empty x_coords at iteration {iteration}, skipping visualization")
                            return
                    else:
                        x_plot = x_coords

                    # 验证数据有效性
                    if x_plot.size == 0 or U_cpu.size == 0:
                        print(f"  Warning: Empty data at iteration {iteration}, skipping visualization")
                        return

                    self.plot_iteration_step(iteration, residual, U_cpu, x_plot)
                except Exception as e:
                    print(f"  Warning: Visualization failed at iteration {iteration}: {e}")
                    # 继续执行，不中断求解过程

    def plot_iteration_step(self, iteration: int, residual: float, U: np.ndarray, x: np.ndarray) -> None:
        """绘制单次迭代步的状态"""
        # 首次调用时创建图窗
        if not self._realtime_mode_active:
            self._init_realtime_plot()

        if self.n_dim == 1:
            self._plot_iteration_step_1d(iteration, residual, U, x)
        else:
            self._plot_iteration_step_2d(iteration, residual, U, x)

    def _init_realtime_plot(self):
        """初始化实时绘图窗口"""
        if self._realtime_mode_active:
            return

        plt.ion()  # 开启交互模式

        # 创建双子图：左边显示U的演化，右边显示残差
        self.fig = self._create_figure(figsize=(14, 5))
        self.ax_solution = self.fig.add_subplot(121)
        self.ax_residual = self.fig.add_subplot(122)
        self.fig.tight_layout(pad=2.0)

        self._realtime_mode_active = True
        print("Real-time visualization window initialized")

    def _plot_iteration_step_1d(self, iteration: int, residual: float, U: np.ndarray, x: np.ndarray) -> None:
        """1D情况的迭代步可视化"""
        # 左图：解的分布
        if self._scatter_solution is None:
            self.ax_solution.clear()
            x_flat = x[:, 0] if x.ndim > 1 else x
            u_flat = U[:, 0] if U.ndim > 1 else U
            self._scatter_solution = self.ax_solution.scatter(x_flat, u_flat, c='b', s=20)
            self.ax_solution.set_xlabel('x')
            self.ax_solution.set_ylabel('u')
            self.ax_solution.grid(True, alpha=0.3)
        else:
            x_flat = x[:, 0] if x.ndim > 1 else x
            u_flat = U[:, 0] if U.ndim > 1 else U
            self._scatter_solution.set_offsets(np.column_stack((x_flat, u_flat)))

        # 动态调整y轴范围
        u_flat = U[:, 0] if U.ndim > 1 else U
        u_min, u_max = u_flat.min(), u_flat.max()
        if u_max > u_min:
            margin = (u_max - u_min) * 0.1
            self.ax_solution.set_ylim(u_min - margin, u_max + margin)

        self.ax_solution.set_title(f'Solution at Iteration {iteration}')

        # 右图：残差历史
        if self._line_residual is None:
            self._line_residual, = self.ax_residual.semilogy(
                self.iteration_history, self.residual_history, 'r-', linewidth=2
            )
            self.ax_residual.set_xlabel('Iteration')
            self.ax_residual.set_ylabel('Residual')
            self.ax_residual.set_title('Residual History')
            self.ax_residual.grid(True, alpha=0.3)
        else:
            self._line_residual.set_data(self.iteration_history, self.residual_history)
            # 更新x轴范围
            if len(self.iteration_history) > 0:
                self.ax_residual.set_xlim(0, max(self.iteration_history) * 1.1)
            # 更新y轴范围
            if len(self.residual_history) > 0:
                y_min = min(self.residual_history)
                y_max = max(self.residual_history)
                if y_max > y_min:
                    self.ax_residual.set_ylim(y_min * 0.5, y_max * 2.0)

        # 刷新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        import time
        time.sleep(0.01)

    def _plot_iteration_step_2d(self, iteration: int, residual: float, U: np.ndarray, x: np.ndarray) -> None:
        """2D情况的迭代步可视化"""
        # 验证数据维度
        if x.shape[1] < 2:
            print(f"  Warning: x has insufficient dimensions ({x.shape}), expected at least 2D")
            return
        if x.shape[0] == 0 or U.shape[0] == 0:
            print(f"  Warning: Empty data arrays (x: {x.shape}, U: {U.shape})")
            return

        # 左图：解的分布（2D散点图）
        self.ax_solution.clear()
        u_vals = U[:, 0] if U.ndim > 1 else U

        # 使用 try-except 捕获 matplotlib 内部错误
        try:
            scatter = self.ax_solution.scatter(x[:, 0], x[:, 1], c=u_vals, cmap='viridis', s=15, alpha=0.8)
            if scatter is not None and hasattr(self.fig, 'colorbar'):
                # 添加颜色条（如果还没有）
                if not hasattr(self, '_colorbar') or self._colorbar is None:
                    self._colorbar = self.fig.colorbar(scatter, ax=self.ax_solution)
        except (StopIteration, ValueError) as e:
            print(f"  Warning: Scatter plot failed: {e}, using simple plot instead")
            # 降级到简单的2D投影
            self.ax_solution.plot(x[:, 0], u_vals, 'b.', markersize=2)

        self.ax_solution.set_xlabel('x')
        self.ax_solution.set_ylabel('t')
        self.ax_solution.set_title(f'Solution at Iteration {iteration}')

        # 右图：残差历史
        if self._line_residual is None:
            self._line_residual, = self.ax_residual.semilogy(
                self.iteration_history, self.residual_history, 'r-', linewidth=2
            )
            self.ax_residual.set_xlabel('Iteration')
            self.ax_residual.set_ylabel('Residual')
            self.ax_residual.set_title('Residual History')
            self.ax_residual.grid(True, alpha=0.3)
        else:
            self._line_residual.set_data(self.iteration_history, self.residual_history)
            if len(self.iteration_history) > 0:
                self.ax_residual.set_xlim(0, max(self.iteration_history) * 1.1)
            if len(self.residual_history) > 0:
                y_min = min(self.residual_history)
                y_max = max(self.residual_history)
                if y_max > y_min:
                    self.ax_residual.set_ylim(y_min * 0.5, y_max * 2.0)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        import time
        time.sleep(0.01)

    def create_iteration_gif(self, filename: str = "iteration_evolution.gif") -> None:
        """创建伪时间步迭代过程的GIF动画"""
        if not self.iteration_history or not self.solution_history:
            print("Warning: 没有迭代数据可创建动画")
            return

        try:
            print(f"正在创建迭代演化动画: {filename}")

            # 清空之前的帧
            self._frames = []

            # 为每次迭代生成帧
            for i, (iteration, residual, solution) in enumerate(zip(
                self.iteration_history, self.residual_history, self.solution_history
            )):
                temp_fig = self._create_figure(figsize=(14, 5))
                ax_sol = temp_fig.add_subplot(121)
                ax_res = temp_fig.add_subplot(122)

                if self.n_dim == 1:
                    # 1D情况：散点图显示解
                    # 需要从data_train获取x坐标
                    if hasattr(self.config, 'x_domain'):
                        n_points = len(solution)
                        x = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], n_points)
                        u_flat = solution.flatten()
                        ax_sol.plot(x, u_flat, 'b-', linewidth=2)
                        ax_sol.set_xlabel('x')
                        ax_sol.set_ylabel('u')
                        ax_sol.set_title(f'Solution at Iteration {iteration}')
                        ax_sol.grid(True, alpha=0.3)

                        # 设置固定的y轴范围
                        all_solutions = np.concatenate([s.flatten() for s in self.solution_history])
                        y_min, y_max = all_solutions.min(), all_solutions.max()
                        margin = (y_max - y_min) * 0.1
                        ax_sol.set_ylim(y_min - margin, y_max + margin)
                else:
                    # 2D情况：热力图
                    u_vals = solution.flatten()
                    side_length = int(np.sqrt(len(u_vals)))
                    if side_length * side_length == len(u_vals):
                        u_grid = u_vals.reshape(side_length, side_length)
                        im = ax_sol.imshow(u_grid, cmap='viridis', aspect='equal')
                        plt.colorbar(im, ax=ax_sol)
                    ax_sol.set_title(f'Solution at Iteration {iteration}')

                # 残差历史
                ax_res.semilogy(self.iteration_history[:i+1], self.residual_history[:i+1],
                               'r-', linewidth=2)
                ax_res.scatter([iteration], [residual], c='r', s=50, zorder=5)
                ax_res.set_xlabel('Iteration')
                ax_res.set_ylabel('Residual')
                ax_res.set_title(f'Residual History (Current: {residual:.2e})')
                ax_res.grid(True, alpha=0.3)

                # 设置固定的x轴范围
                ax_res.set_xlim(0, max(self.iteration_history) * 1.05)
                # 设置固定的y轴范围
                y_min = min(self.residual_history)
                y_max = max(self.residual_history)
                ax_res.set_ylim(y_min * 0.5, y_max * 2.0)

                plt.tight_layout()

                # 将当前帧添加到动画
                buf = io.BytesIO()
                temp_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                self._frames.append(Image.open(buf).copy())
                self._close_figure(temp_fig)

                if (i + 1) % 10 == 0:
                    print(f"  已生成 {i + 1}/{len(self.iteration_history)} 帧")

            # 保存GIF动画
            if len(self._frames) > 0:
                os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
                self._frames[0].save(
                    filename,
                    save_all=True,
                    append_images=self._frames[1:],
                    optimize=True,
                    duration=200,
                    loop=0,
                )
                print(f"迭代演化动画已保存至 {filename}")
                self._frames = []
            else:
                print("没有帧可保存")

        except Exception as e:
            print(f"动画创建失败: {e}")

    def finalize_realtime_animation(self) -> None:
        """完成实时动画显示"""
        if self._realtime_mode_active:
            plt.ioff()  # 关闭交互模式，防止后续创建图窗自动显示
            self._realtime_mode_active = False
            print("Pseudo-time iteration visualization completed.")

    def plot_final_solution(self, U: np.ndarray, data: Dict, save_path: Optional[str] = None) -> None:
        """绘制最终解（测试集上的网格图）"""
        # Ensure U is on CPU
        U = self._to_numpy_cpu(U)

        fig = self._create_figure(figsize=(12, 5))

        if self.n_dim == 1:
            # 获取x坐标
            if 'x_segments' in data:
                x_coords = np.vstack(data['x_segments'])
            elif 'x' in data:
                x_coords = data['x']
            else:
                x_coords = np.linspace(self.config.x_domain[0][0],
                                      self.config.x_domain[0][1], len(U))

            x_flat = x_coords[:, 0] if x_coords.ndim > 1 else x_coords
            u_flat = U.flatten()

            # 排序
            sort_idx = np.argsort(x_flat)
            x_sorted = x_flat[sort_idx]
            u_sorted = u_flat[sort_idx]

            ax = fig.add_subplot(111)
            ax.plot(x_sorted, u_sorted, 'b-', linewidth=2, label='Steady-state Solution')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('u', fontsize=12)
            ax.set_title('Final Steady-State Solution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            # 2D情况：使用网格绘图（test_data是规则网格）
            if 'x_segments' in data:
                x_coords = np.vstack(data['x_segments'])
            elif 'x' in data:
                x_coords = data['x']
            else:
                print("Warning: 无法获取x坐标数据")
                return

            ax = fig.add_subplot(111)
            u_vals = U.flatten()

            y_label = getattr(self.config, 'spatial_vars', ['x', 'y'])[1]

            # 提取唯一值构建网格（和参考解一样的方法）
            x_unique = np.unique(x_coords[:, 0])
            y_unique = np.unique(x_coords[:, 1])

            # 创建网格坐标
            X_mesh, Y_mesh = np.meshgrid(x_unique, y_unique, indexing='xy')

            # 将数据正确映射到网格（考虑分区结构）
            U_grid = self._map_to_grid(x_coords[:, 0], x_coords[:, 1], u_vals,
                                       x_unique, y_unique)

            # 使用 pcolormesh 绘制
            im = ax.pcolormesh(X_mesh, Y_mesh, U_grid, cmap='RdBu_r', shading='gouraud')
            plt.colorbar(im, ax=ax, label='u')

            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title('Final Steady-State Solution', fontsize=14, fontweight='bold')
            ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)
            print(f"最终解已保存至: {save_path}")

        self._close_figure(fig)

    def _map_to_grid(self, x: np.ndarray, y: np.ndarray, u_vals: np.ndarray,
                     x_unique: np.ndarray, y_unique: np.ndarray) -> np.ndarray:
        """将数据正确映射到规则网格（考虑分区结构）

        Args:
            x: x坐标数组 (n_points,)
            y: y坐标数组 (n_points,)
            u_vals: 解值数组 (n_points,)
            x_unique: 唯一x值（已排序）
            y_unique: 唯一y值（已排序）

        Returns:
            U_grid: 映射后的网格数组 (len(y_unique), len(x_unique))
        """
        # 创建网格索引 - 使用searchsorted找到每个点在网格中的位置
        x_indices = np.searchsorted(x_unique, x)
        y_indices = np.searchsorted(y_unique, y)

        # 创建网格数组并填充
        U_grid = np.full((len(y_unique), len(x_unique)), np.nan)
        U_grid[y_indices, x_indices] = u_vals

        # 如果有NaN值（可能是插值问题），使用griddata填充
        if np.any(np.isnan(U_grid)):
            print(f"  Warning: Grid has {np.sum(np.isnan(U_grid))} NaN values, using griddata to fill")
            X_mesh, Y_mesh = np.meshgrid(x_unique, y_unique, indexing='xy')
            x_coords = np.column_stack([x, y])
            U_grid = griddata(x_coords, u_vals, (X_mesh, Y_mesh), method='linear')

            # 如果还有NaN，使用nearest填充
            if np.any(np.isnan(U_grid)):
                nan_mask = np.isnan(U_grid)
                X_flat = X_mesh.flatten()
                Y_flat = Y_mesh.flatten()
                nan_points = np.column_stack([X_flat[nan_mask.flatten()], Y_flat[nan_mask.flatten()]])
                U_grid_flat = U_grid.flatten()
                U_grid_flat[nan_mask.flatten()] = griddata(x_coords, u_vals, nan_points, method='nearest')
                U_grid = U_grid_flat.reshape(len(y_unique), len(x_unique))

        return U_grid

    def _try_reshape_to_grid(self, x_coords: np.ndarray, u_vals: np.ndarray):
        """尝试将数据重塑为规则网格

        Args:
            x_coords: 坐标数组 (n_points, n_dim)
            u_vals: 解值数组 (n_points,)

        Returns:
            is_grid: 是否为规则网格
            X_grid: X网格（如果是规则网格）
            Y_grid: Y网格（如果是规则网格）
            U_grid: U网格（如果是规则网格）
        """
        if x_coords.shape[1] < 2:
            return False, None, None, None

        x = x_coords[:, 0]
        y = x_coords[:, 1]

        # 提取唯一值
        x_unique = np.unique(x)
        y_unique = np.unique(y)

        # 检查是否为规则网格
        expected_size = len(x_unique) * len(y_unique)
        if len(x) != expected_size:
            return False, None, None, None

        # 尝试重塑
        try:
            U_grid = self._map_to_grid(x, y, u_vals, x_unique, y_unique)

            # 创建网格坐标
            X_grid, Y_grid = np.meshgrid(x_unique, y_unique, indexing='xy')

            return True, X_grid, Y_grid, U_grid

        except:
            return False, None, None, None

    def plot_residual_history(self, save_path: Optional[str] = None) -> None:
        """绘制残差历史"""
        if not self.iteration_history or not self.residual_history:
            print("Warning: 没有残差历史数据")
            return

        fig = self._create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        ax.semilogy(self.iteration_history, self.residual_history, 'r-', linewidth=2,
                   marker='o', markersize=3)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Residual (log scale)', fontsize=12)
        ax.set_title('Pseudo-Time Stepping Residual History', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        if self.residual_history:
            final_residual = self.residual_history[-1]
            initial_residual = self.residual_history[0]
            ax.text(0.02, 0.98,
                   f'Initial Residual: {initial_residual:.2e}\nFinal Residual: {final_residual:.2e}\nReduction: {initial_residual/final_residual:.2e}x',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)
            print(f"残差历史已保存至: {save_path}")

        self._close_figure(fig)

    def close_evolution_plot(self) -> None:
        """关闭演化绘图"""
        if self._realtime_mode_active:
            plt.ioff()
            self._realtime_mode_active = False
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def save_convergence_history(self, conv_info: Dict, save_dir: str) -> None:
        """保存收敛历史图表

        Args:
            conv_info: 包含收敛信息的字典
            save_dir: 保存目录
        """
        # 确保在非交互模式下创建图窗（仅保存，不显示）
        was_interactive = plt.isinteractive()
        if was_interactive:
            plt.ioff()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Residual history
        axes[0, 0].semilogy(conv_info["residual_history"])
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Residual")
        axes[0, 0].set_title("Convergence History")
        axes[0, 0].grid(True, alpha=0.3)

        # Time step history
        axes[0, 1].plot(conv_info["dt_history"])
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("dt (pseudo-time step)")
        axes[0, 1].set_title("Adaptive Time Step")
        axes[0, 1].grid(True, alpha=0.3)

        # Solution norm history
        axes[1, 0].plot(conv_info["solution_norm_history"])
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("||U||")
        axes[1, 0].set_title("Solution Norm Evolution")
        axes[1, 0].grid(True, alpha=0.3)

        # Convergence rate
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

        # 恢复交互模式状态
        if was_interactive:
            plt.ion()

        print(f"收敛历史已保存至: {save_path}")

    def visualize_solution(self, U_steady: np.ndarray, data: Dict, save_dir: str) -> None:
        """可视化稳态解（测试集上的网格图）

        Args:
            U_steady: 稳态解
            data: 数据字典
            save_dir: 保存目录
        """
        # Ensure U_steady is on CPU
        U_steady = self._to_numpy_cpu(U_steady)

        x = data['x']
        n_eqs = U_steady.shape[1] if U_steady.ndim > 1 else 1

        if self.n_dim == 1:
            # 1D plot
            fig, axes = plt.subplots(1, n_eqs, figsize=(6*n_eqs, 5))
            if n_eqs == 1:
                axes = [axes]

            for eq_idx in range(n_eqs):
                u_vals = U_steady[:, eq_idx] if U_steady.ndim > 1 else U_steady
                x_flat = x[:, 0] if x.ndim > 1 else x

                # 排序
                sort_idx = np.argsort(x_flat)
                x_sorted = x_flat[sort_idx]
                u_sorted = u_vals[sort_idx]

                axes[eq_idx].plot(x_sorted, u_sorted, 'b-', linewidth=2, label='Solution')
                axes[eq_idx].set_xlabel('x', fontsize=12)
                axes[eq_idx].set_ylabel(f'u_{eq_idx}', fontsize=12)
                axes[eq_idx].set_title(f'Steady-State Solution (Equation {eq_idx})', fontsize=13, fontweight='bold')
                axes[eq_idx].grid(True, alpha=0.3)
                axes[eq_idx].legend()

            plt.tight_layout()
            save_path = os.path.join(save_dir, "steady_state_solution.png")
            self._save_figure(fig, save_path)
            self._close_figure(fig)

        elif self.n_dim == 2:
            # 2D网格图（test_data是规则网格，采用和参考解一样的方法）
            y_label = getattr(self.config, 'spatial_vars', ['x', 'y'])[1]

            # 提取唯一值构建网格
            x_unique = np.unique(x[:, 0])
            y_unique = np.unique(x[:, 1])

            # 创建网格坐标
            X_mesh, Y_mesh = np.meshgrid(x_unique, y_unique, indexing='xy')

            for eq_idx in range(n_eqs):
                fig = self._create_figure(figsize=(10, 8))
                ax = fig.add_subplot(111)
                u_vals = U_steady[:, eq_idx] if U_steady.ndim > 1 else U_steady

                # 将数据正确映射到网格（考虑分区结构）
                U_grid = self._map_to_grid(x[:, 0], x[:, 1], u_vals, x_unique, y_unique)

                # 使用 pcolormesh 绘制
                im = ax.pcolormesh(X_mesh, Y_mesh, U_grid, cmap='RdBu_r', shading='gouraud')
                plt.colorbar(im, ax=ax, label=f'u_{eq_idx}')

                ax.set_xlabel('x', fontsize=12)
                ax.set_ylabel(y_label, fontsize=12)
                ax.set_title(f'Steady-State Solution (Equation {eq_idx})', fontsize=14, fontweight='bold')
                ax.set_aspect('equal')
                plt.tight_layout()

                save_path = os.path.join(save_dir, f"steady_state_solution_eq{eq_idx}.png")
                self._save_figure(fig, save_path)
                self._close_figure(fig)

    def plot_reference_comparison(self, U_steady: np.ndarray, data: Dict,
                                   ref_solution: Dict, save_path: str) -> None:
        """绘制数值解与参考解的对比图，包括残差分布

        Args:
            U_steady: 稳态数值解
            data: 训练数据字典
            ref_solution: 参考解字典
            save_path: 保存路径
        """
        # Ensure U_steady is on CPU
        U_steady = self._to_numpy_cpu(U_steady)

        # 获取训练点坐标
        x_coords = data['x']
        U_flat = U_steady.flatten()

        if self.n_dim == 1:
            # 1D问题的参考解对比
            self._plot_reference_comparison_1d(x_coords, U_flat, ref_solution, save_path)
        elif self.n_dim == 2:
            # 2D问题的参考解对比（类似time_pde的spacetime_comparison）
            self._plot_reference_comparison_2d(x_coords, U_flat, ref_solution, save_path)
        else:
            print(f"Warning: {self.n_dim}D reference comparison not supported")

    def _plot_reference_comparison_1d(self, x_coords: np.ndarray, U_flat: np.ndarray,
                                       ref_solution: Dict, save_path: str) -> None:
        """1D问题的参考解对比"""
        x_flat = x_coords[:, 0] if x_coords.ndim > 1 else x_coords
        x_ref = ref_solution['x_ref']
        u_ref = ref_solution['u_ref']

        # 插值参考解到训练点
        interp_func = interp1d(x_ref, u_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
        u_ref_interp = interp_func(x_flat)

        # 计算误差
        error = np.abs(U_flat - u_ref_interp)

        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # 上图：解的对比
        axes[0].plot(x_ref, u_ref, 'b-', linewidth=2, label='Reference Solution', alpha=0.8)
        axes[0].scatter(x_flat, U_flat, c='red', s=30, alpha=0.8,
                       label='Numerical Solution', zorder=5)
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel('u', fontsize=12)
        axes[0].set_title('Solution Comparison', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # 下图：误差分布
        axes[1].scatter(x_flat, error, c='red', s=25, alpha=0.8)
        axes[1].set_xlabel('x', fontsize=12)
        axes[1].set_ylabel('|Error|', fontsize=12)
        axes[1].set_title('Pointwise Error (Log Scale)', fontsize=14, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        # 添加误差统计
        max_error = np.max(error)
        l2_error = np.sqrt(np.mean(error**2))
        axes[1].text(0.02, 0.98, f'Max Error: {max_error:.2e}\\nL2 Error: {l2_error:.2e}',
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
        print(f"参考解对比已保存至: {save_path}")
        print(f"  Max Error: {max_error:.2e}, L2 Error: {l2_error:.2e}")

    def _plot_reference_comparison_2d(self, x_coords: np.ndarray, U_flat: np.ndarray,
                                       ref_solution: Dict, save_path: str) -> None:
        """2D问题的参考解对比（三子图布局：数值解、参考解、误差）"""
        # 测试点坐标（应该是规则网格）
        x_test = x_coords[:, 0]
        y_test = x_coords[:, 1]

        # 参考解坐标和值
        x_ref = ref_solution['x_ref']
        y_ref = ref_solution.get('y_ref', None)
        u_ref = ref_solution['u_ref']
        y_label = ref_solution.get('y_label', 'y')  # Get axis label for second dimension

        # 插值参考解到测试点以计算误差
        if y_ref is not None:
            # 2D参考解：创建网格点进行插值
            # 参考数据网格：(y, x) 或 (t, x) 的笛卡尔积
            X_ref_grid, Y_ref_grid = np.meshgrid(x_ref, y_ref, indexing='xy')
            points_ref = np.column_stack([X_ref_grid.flatten(), Y_ref_grid.flatten()])

            # 测试点
            points_test = np.column_stack([x_test, y_test])

            # 插值
            u_ref_interp = griddata(points_ref, u_ref, points_test, method='linear')

            # 处理可能的 NaN 值（边界外的点）
            nan_mask = np.isnan(u_ref_interp)
            if np.any(nan_mask):
                print(f"  Warning: {np.sum(nan_mask)} points outside reference domain, using nearest interpolation")
                u_ref_interp[nan_mask] = griddata(points_ref, u_ref, points_test[nan_mask], method='nearest')
        else:
            # 1D参考解扩展到2D（假设参考解只依赖x）
            interp_func = interp1d(x_ref, u_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
            u_ref_interp = interp_func(x_test)

        # 计算逐点误差
        error = np.abs(U_flat - u_ref_interp)
        error_nonzero = np.maximum(error, 1e-12)  # 避免log(0)

        # 创建三子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 获取全局颜色范围
        u_min = min(np.nanmin(U_flat), np.nanmin(u_ref))
        u_max = max(np.nanmax(U_flat), np.nanmax(u_ref))

        # 子图1: 数值解（测试集网格图，采用和参考解一样的画法）
        # 提取唯一值构建网格
        x_unique = np.unique(x_test)
        y_unique = np.unique(y_test)

        # 创建网格坐标
        X_test_mesh, Y_test_mesh = np.meshgrid(x_unique, y_unique, indexing='xy')

        # 将数据正确映射到网格（考虑分区结构）
        U_test_grid = self._map_to_grid(x_test, y_test, U_flat, x_unique, y_unique)

        # 使用 pcolormesh 绘制（和参考解一样的方法）
        im1 = axes[0].pcolormesh(X_test_mesh, Y_test_mesh, U_test_grid, cmap='RdBu_r',
                                 shading='gouraud', vmin=u_min, vmax=u_max)

        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel(y_label, fontsize=12)
        axes[0].set_title('Numerical Solution (Test Grid)', fontsize=12, fontweight='bold')
        axes[0].set_aspect('equal')
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('u', fontsize=11)

        # 子图2: 参考解（高分辨率）
        if y_ref is not None:
            # 创建网格用于pcolormesh
            X_ref_mesh, Y_ref_mesh = np.meshgrid(x_ref, y_ref, indexing='xy')
            # 重塑 u_ref 为 (n_y, n_x) 网格
            u_ref_grid = u_ref.reshape(len(y_ref), len(x_ref))
            im2 = axes[1].pcolormesh(X_ref_mesh, Y_ref_mesh, u_ref_grid, cmap='RdBu_r',
                                    shading='gouraud', vmin=u_min, vmax=u_max)
        else:
            # 如果没有y_ref，使用散点图
            scatter2 = axes[1].scatter(x_ref, np.zeros_like(x_ref), c=u_ref,
                                      cmap='RdBu_r', s=20, vmin=u_min, vmax=u_max)
            im2 = scatter2

        axes[1].set_xlabel('x', fontsize=12)
        axes[1].set_ylabel(y_label, fontsize=12)
        axes[1].set_title('Reference Solution (High-Res)', fontsize=12, fontweight='bold')
        axes[1].set_aspect('equal')
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('u', fontsize=11)

        # 子图3: 逐点误差（对数刻度，采用和参考解一样的画法）
        error_p05 = np.percentile(error_nonzero, 5)
        error_p95 = np.percentile(error_nonzero, 95)
        vmin_error = max(error_p05, 1e-10)
        vmax_error = min(error_p95 * 5, np.max(error_nonzero))

        # 将误差数据正确映射到网格（考虑分区结构）
        Error_grid = self._map_to_grid(x_test, y_test, error_nonzero, x_unique, y_unique)

        # 使用 pcolormesh 绘制误差
        im3 = axes[2].pcolormesh(X_test_mesh, Y_test_mesh, Error_grid, cmap='plasma',
                                 shading='gouraud', norm=LogNorm(vmin=vmin_error, vmax=vmax_error))

        axes[2].set_xlabel('x', fontsize=12)
        axes[2].set_ylabel(y_label, fontsize=12)
        axes[2].set_title('Pointwise Error (Log Scale)', fontsize=12, fontweight='bold')
        axes[2].set_aspect('equal')
        cbar3 = plt.colorbar(im3, ax=axes[2])
        cbar3.set_label('|Error|', fontsize=11)

        # 添加误差统计
        max_error = np.nanmax(error)
        l2_error = np.sqrt(np.nanmean(error**2))
        median_error = np.nanmedian(error)
        axes[2].text(0.02, 0.98,
                    f'Max: {max_error:.2e}\\nL2: {l2_error:.2e}\\nMedian: {median_error:.2e}',
                    transform=axes[2].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

        print(f"参考解对比已保存至: {save_path}")
        print(f"  Max Error: {max_error:.2e}, L2 Error: {l2_error:.2e}, Median Error: {median_error:.2e}")

        # 添加 t=1 时刻的切片对比图（仅针对2D问题）
        if y_ref is not None:
            self._plot_t1_slice_comparison(x_coords, U_flat, ref_solution, save_path)

    def _plot_t1_slice_comparison(self, x_coords: np.ndarray, U_flat: np.ndarray,
                                   ref_solution: Dict, base_save_path: str) -> None:
        """绘制 t=1 时刻的切片对比图（参考解用线图，数值解用散点）

        Args:
            x_coords: 训练点坐标 (n_points, 2)
            U_flat: 数值解 (n_points,)
            ref_solution: 参考解字典
            base_save_path: 基础保存路径
        """
        x_train = x_coords[:, 0]
        y_train = x_coords[:, 1]  # 实际是 t

        x_ref = ref_solution['x_ref']
        y_ref = ref_solution['y_ref']  # 实际是 t_ref
        u_ref = ref_solution['u_ref']
        y_label = ref_solution.get('y_label', 'y')

        # 找到 t=1 (y=1) 附近的数据点
        t_target = 1.0

        # 从数值解中提取 t≈1 的点（test_data是规则网格）
        # 找到最接近 t=1 的 y 值
        y_unique = np.unique(y_train)
        t_idx_numerical = np.argmin(np.abs(y_unique - t_target))
        t_actual_numerical = y_unique[t_idx_numerical]

        # 提取该时刻的所有点
        mask_numerical = np.abs(y_train - t_actual_numerical) < 1e-10

        if not np.any(mask_numerical):
            print(f"  Warning: No numerical solution points at t={t_actual_numerical}")
            return

        x_numerical_t1 = x_train[mask_numerical]
        u_numerical_t1 = U_flat[mask_numerical]

        # 对数值解按 x 排序以便可视化
        sort_idx = np.argsort(x_numerical_t1)
        x_numerical_t1 = x_numerical_t1[sort_idx]
        u_numerical_t1 = u_numerical_t1[sort_idx]

        # 从参考解中提取 t=1 的切片
        # 找到最接近 t=1 的 y_ref 索引
        t_idx = np.argmin(np.abs(y_ref - t_target))
        t_actual = y_ref[t_idx]

        # 参考解的形状是 (len(y_ref), len(x_ref))
        u_ref_grid = u_ref.reshape(len(y_ref), len(x_ref))
        u_ref_t1 = u_ref_grid[t_idx, :]  # 提取 t=t_actual 时刻的切片

        # 创建对比图
        fig = self._create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # 参考解：线图（蓝色实线）
        ax.plot(x_ref, u_ref_t1, 'b-', linewidth=2, label=f'Reference Solution ({y_label}={t_actual:.3f})', alpha=0.8)

        # 数值解：线图（红色虚线）
        ax.plot(x_numerical_t1, u_numerical_t1, 'r--', linewidth=2,
                label=f'Numerical Solution ({y_label}={t_actual_numerical:.3f})', alpha=0.8, zorder=5)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u', fontsize=12)
        ax.set_title(f'Solution Comparison at {y_label}={t_target}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 计算该切片的误差统计
        # 插值参考解到数值解的 x 坐标
        from scipy.interpolate import interp1d
        interp_func = interp1d(x_ref, u_ref_t1, kind='cubic', bounds_error=False, fill_value='extrapolate')
        u_ref_interp = interp_func(x_numerical_t1)

        error_t1 = np.abs(u_numerical_t1 - u_ref_interp)
        max_error_t1 = np.max(error_t1)
        l2_error_t1 = np.sqrt(np.mean(error_t1**2))

        # 添加误差统计文本框
        ax.text(0.02, 0.98,
               f'Max Error: {max_error_t1:.2e}\\nL2 Error: {l2_error_t1:.2e}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10)

        plt.tight_layout()

        # 保存图形
        save_dir = os.path.dirname(base_save_path)
        t1_save_path = os.path.join(save_dir, f"{y_label}1_slice_comparison.png")
        self._save_figure(fig, t1_save_path)
        self._close_figure(fig)

        print(f"{y_label}=1 切片对比已保存至: {t1_save_path}")
        print(f"  切片误差 - Max: {max_error_t1:.2e}, L2: {l2_error_t1:.2e}")

    def save_solution_data(self, U_train: np.ndarray, U_test: np.ndarray,
                           coeffs_steady: np.ndarray, conv_info: Dict,
                           data_train: Dict, data_test: Dict, save_dir: str) -> None:
        """保存解数据到文件（训练和测试集）

        Args:
            U_train: 训练集上的稳态解
            U_test: 测试集上的稳态解
            coeffs_steady: 稳态系数
            conv_info: 收敛信息
            data_train: 训练数据字典
            data_test: 测试数据字典
            save_dir: 保存目录
        """
        # Ensure all data is on CPU before saving
        U_train = self._to_numpy_cpu(U_train)
        U_test = self._to_numpy_cpu(U_test)
        coeffs_steady = self._to_numpy_cpu(coeffs_steady)

        save_path = os.path.join(save_dir, "solution_data.npz")
        np.savez(
            save_path,
            x_train=data_train['x'],
            x_test=data_test['x'],
            U_train=U_train,
            U_test=U_test,
            coeffs=coeffs_steady,
            residual_history=conv_info["residual_history"],
            dt_history=conv_info["dt_history"],
            iterations=conv_info["iterations"],
            converged=conv_info["converged"]
        )
        print(f"✓ Solution data saved: {save_path}")