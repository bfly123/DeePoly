import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from PIL import Image
import io
import os
from src.abstract_class.config.base_visualize import BaseVisualizer


class GeneralPDEVisualizer(BaseVisualizer):
    """Visualizer for general PDE problems with pseudo-time stepping

    Extends BaseVisualizer with specific functionality for:
    - Pseudo-time stepping iteration visualization
    - Residual convergence monitoring
    - Dual-panel displays (solution + residual)
    """

    def __init__(self, config):
        super().__init__(config)
        self.n_dim = len(config.spatial_vars) if hasattr(config, 'spatial_vars') else len(config.x_domain)

        plt.ion()

        # 创建双子图：左边显示U的演化，右边显示残差
        self.fig = self._create_figure(figsize=(14, 5))
        self.ax_solution = self.fig.add_subplot(121)
        self.ax_residual = self.fig.add_subplot(122)

        self.fig.tight_layout(pad=2.0)

        # 用于存储动画帧
        self._frames = []
        self._save_gif = False

        # 用于实时更新的图形对象
        self._scatter_solution = None
        self._line_residual = None
        self._colorbar = None  # 用于2D可视化的颜色条

        # 历史数据
        self.iteration_history = []
        self.residual_history = []
        self.solution_history = []

    def reset_animation_data(self):
        """重置动画数据"""
        self.iteration_history = []
        self.residual_history = []
        self.solution_history = []
        self._frames = []
        print("动画数据已重置")

    def update_animation_data(self, iteration: int, residual: float, U: np.ndarray,
                            data_train: Dict, animation_skip: int = 1) -> None:
        """更新动画数据"""
        if iteration % animation_skip == 0:
            self.iteration_history.append(iteration)
            self.residual_history.append(residual)
            self.solution_history.append(U.copy())

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
                    if x_plot.size == 0 or U.size == 0:
                        print(f"  Warning: Empty data at iteration {iteration}, skipping visualization")
                        return

                    self.plot_iteration_step(iteration, residual, U, x_plot)
                except Exception as e:
                    print(f"  Warning: Visualization failed at iteration {iteration}: {e}")
                    # 继续执行，不中断求解过程

    def plot_iteration_step(self, iteration: int, residual: float, U: np.ndarray, x: np.ndarray) -> None:
        """绘制单次迭代步的状态"""
        if self.n_dim == 1:
            self._plot_iteration_step_1d(iteration, residual, U, x)
        else:
            self._plot_iteration_step_2d(iteration, residual, U, x)

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
        plt.ioff()
        print("Pseudo-time iteration visualization completed. Close the plot window to continue...")

    def plot_final_solution(self, U: np.ndarray, data: Dict, save_path: Optional[str] = None) -> None:
        """绘制最终解"""
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
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title('Final Steady-State Solution')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            # 2D情况
            if 'x_segments' in data:
                x_coords = np.vstack(data['x_segments'])
            elif 'x' in data:
                x_coords = data['x']
            else:
                print("Warning: 无法获取x坐标数据")
                return

            ax = fig.add_subplot(111)
            u_vals = U.flatten()
            scatter = ax.scatter(x_coords[:, 0], x_coords[:, 1], c=u_vals,
                               cmap='viridis', s=20)
            plt.colorbar(scatter, ax=ax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Final Steady-State Solution')
            ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)
            print(f"最终解已保存至: {save_path}")

        self._close_figure(fig)

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
        plt.ioff()
        if hasattr(self, 'fig'):
            plt.close(self.fig)

    def save_convergence_history(self, conv_info: Dict, save_dir: str) -> None:
        """保存收敛历史图表

        Args:
            conv_info: 包含收敛信息的字典
            save_dir: 保存目录
        """
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
        print(f"收敛历史已保存至: {save_path}")

    def visualize_solution(self, U_steady: np.ndarray, data: Dict, save_dir: str) -> None:
        """可视化稳态解

        Args:
            U_steady: 稳态解
            data: 数据字典
            save_dir: 保存目录
        """
        x = data['x']
        n_eqs = U_steady.shape[1] if U_steady.ndim > 1 else 1

        if self.n_dim == 1:
            # 1D plot
            fig, axes = plt.subplots(1, n_eqs, figsize=(6*n_eqs, 5))
            if n_eqs == 1:
                axes = [axes]

            for eq_idx in range(n_eqs):
                u_vals = U_steady[:, eq_idx] if U_steady.ndim > 1 else U_steady
                axes[eq_idx].plot(x, u_vals, 'b-', linewidth=2, label='Solution')
                axes[eq_idx].set_xlabel('x')
                axes[eq_idx].set_ylabel(f'u_{eq_idx}')
                axes[eq_idx].set_title(f'Steady-State Solution (Equation {eq_idx})')
                axes[eq_idx].grid(True, alpha=0.3)
                axes[eq_idx].legend()

            plt.tight_layout()
            save_path = os.path.join(save_dir, "steady_state_solution.png")
            self._save_figure(fig, save_path)
            self._close_figure(fig)

        elif self.n_dim == 2:
            # 2D contour plot
            for eq_idx in range(n_eqs):
                fig = self._create_figure(figsize=(8, 6))
                u_vals = U_steady[:, eq_idx] if U_steady.ndim > 1 else U_steady
                plt.scatter(x[:, 0], x[:, 1], c=u_vals, cmap='viridis')
                plt.colorbar(label=f'u_{eq_idx}')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'Steady-State Solution (Equation {eq_idx})')
                plt.tight_layout()
                save_path = os.path.join(save_dir, f"steady_state_solution_eq{eq_idx}.png")
                self._save_figure(fig, save_path)
                self._close_figure(fig)

    def save_solution_data(self, U_steady: np.ndarray, coeffs_steady: np.ndarray,
                           conv_info: Dict, data: Dict, save_dir: str) -> None:
        """保存解数据到文件

        Args:
            U_steady: 稳态解
            coeffs_steady: 稳态系数
            conv_info: 收敛信息
            data: 数据字典
            save_dir: 保存目录
        """
        save_path = os.path.join(save_dir, "solution_data.npz")
        np.savez(
            save_path,
            x=data['x'],
            U_steady=U_steady,
            coeffs=coeffs_steady,
            residual_history=conv_info["residual_history"],
            dt_history=conv_info["dt_history"],
            iterations=conv_info["iterations"],
            converged=conv_info["converged"]
        )
        print(f"✓ Solution data saved: {save_path}")