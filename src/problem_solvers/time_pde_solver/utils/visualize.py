import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from src.abstract_class.config.base_visualize import BaseVisualizer
import torch
from PIL import Image
import io
import os
from matplotlib import animation

class TimePDEVisualizer(BaseVisualizer):
    """Visualizer for time-dependent PDE problems - supports independent 1D and 2D visualization"""
    
    def __init__(self, config):
        super().__init__(config)
        # Detect problem dimension
        self.n_dim = len(config.spatial_vars) if hasattr(config, 'spatial_vars') else len(config.x_domain)
        self.is_1d = (self.n_dim == 1)
        
        # 初始化交互式绘图
        plt.ion()
        
        # 根据维度初始化不同的图形
        if self.is_1d:
            self.fig = plt.figure(figsize=(12, 6))
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection="3d")
            
        self._frames = []  # 用于存储动画帧
        self._save_gif = False
        self._visualization_interval = 1
        self._iteration_counter = 0
        self.cbar = None
        
    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """绘制解 - 根据维度自动选择1D或2D可视化"""
        if self.is_1d:
            self._plot_solution_1d(data, prediction, save_path)
        else:
            self._plot_solution_2d(data, prediction, save_path)
    
    def _plot_solution_1d(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """1D解的可视化"""
        fig = self._create_figure(figsize=(14, 5))
        
        # 获取全局解
        x_global = np.vstack(data['x_segments'])
        u_global = np.vstack(data['u_segments'])
        pred_global = prediction
        
        # 排序以便正确绘制线图
        x = x_global[:, 0]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        u_sorted = u_global.flatten()[sort_idx]
        pred_sorted = pred_global.flatten()[sort_idx]
        
        # 创建三个子图：真实解、预测解、误差
        ax1 = fig.add_subplot(131)
        ax1.plot(x_sorted, u_sorted, 'b-', linewidth=2, label='真实解')
        ax1.set_title('真实解')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2 = fig.add_subplot(132)
        ax2.plot(x_sorted, pred_sorted, 'r-', linewidth=2, label='预测解')
        ax2.set_title('预测解')
        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 对比图
        ax3 = fig.add_subplot(133)
        ax3.plot(x_sorted, u_sorted, 'b-', linewidth=2, label='真实解', alpha=0.7)
        ax3.plot(x_sorted, pred_sorted, 'r--', linewidth=2, label='预测解', alpha=0.7)
        ax3.set_title('对比')
        ax3.set_xlabel('x')
        ax3.set_ylabel('u(x)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 添加段边界标记
        if hasattr(self.config, 'n_segments') and hasattr(self.config, 'x_domain'):
            x_min, x_max = self.config.x_domain[0]
            n_seg = self.config.n_segments[0] if isinstance(self.config.n_segments, (list, tuple)) else self.config.n_segments
            segment_boundaries = np.linspace(x_min, x_max, n_seg + 1)
            
            for ax in [ax1, ax2, ax3]:
                for boundary in segment_boundaries[1:-1]:  # 跳过首尾边界
                    ax.axvline(boundary, color='k', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
    
    def _plot_solution_2d(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """2D解的可视化"""
        fig = self._create_figure(figsize=(12, 5))
        
        # 获取全局解
        x_global = np.vstack(data['x_segments'])
        u_global = np.vstack(data['u_segments'])
        pred_global = prediction
        
        # 创建网格
        x = np.unique(x_global[:, 0])
        y = np.unique(x_global[:, 1])
        X, Y = np.meshgrid(x, y)
        
        # 重塑预测结果
        pred_grid = pred_global.reshape(len(y), len(x))
        u_grid = u_global.reshape(len(y), len(x))
        
        # 真实解
        ax1 = fig.add_subplot(121)
        im1 = ax1.pcolormesh(X, Y, u_grid, cmap='viridis')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('真实解')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        
        # 预测解
        ax2 = fig.add_subplot(122)
        im2 = ax2.pcolormesh(X, Y, pred_grid, cmap='viridis')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('预测解')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        
        # 添加段边界
        self._add_segment_boundaries_2d([ax1, ax2])
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
        
    def plot_error(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """绘制误差分布 - 根据维度自动选择1D或2D可视化"""
        if self.is_1d:
            self._plot_error_1d(data, prediction, save_path)
        else:
            self._plot_error_2d(data, prediction, save_path)
    
    def _plot_error_1d(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """1D误差分布可视化"""
        fig = self._create_figure(figsize=(12, 8))
        
        # 计算误差
        x_global = np.vstack(data['x_segments'])
        u_global = np.vstack(data['u_segments'])
        error = np.abs(prediction - u_global)
        relative_error = error / (np.abs(u_global) + 1e-12)  # 避免除零
        
        # 排序
        x = x_global[:, 0]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        error_sorted = error.flatten()[sort_idx]
        rel_error_sorted = relative_error.flatten()[sort_idx]
        u_sorted = u_global.flatten()[sort_idx]
        pred_sorted = prediction.flatten()[sort_idx]
        
        # 创建四个子图
        ax1 = fig.add_subplot(221)
        ax1.plot(x_sorted, error_sorted, 'r-', linewidth=2, label='Absolute Error')
        ax1.set_title('Absolute Error Distribution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('|u_true - u_pred|')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        ax2 = fig.add_subplot(222)
        ax2.plot(x_sorted, rel_error_sorted, 'g-', linewidth=2, label='Relative Error')
        ax2.set_title('Relative Error Distribution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('|u_true - u_pred|/|u_true|')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 误差统计
        ax3 = fig.add_subplot(223)
        ax3.semilogy(x_sorted, np.abs(u_sorted), 'b-', linewidth=2, label='|真实解|', alpha=0.7)
        ax3.semilogy(x_sorted, np.abs(pred_sorted), 'r--', linewidth=2, label='|预测解|', alpha=0.7)
        ax3.set_title('解的量级对比')
        ax3.set_xlabel('x')
        ax3.set_ylabel('|u|')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 误差直方图
        ax4 = fig.add_subplot(224)
        ax4.hist(np.log10(error_sorted + 1e-16), bins=30, alpha=0.7, edgecolor='black')
        ax4.set_title('Error Distribution Histogram')
        ax4.set_xlabel('log10(Absolute Error)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 添加段边界标记
        if hasattr(self.config, 'n_segments') and hasattr(self.config, 'x_domain'):
            x_min, x_max = self.config.x_domain[0]
            n_seg = self.config.n_segments[0] if isinstance(self.config.n_segments, (list, tuple)) else self.config.n_segments
            segment_boundaries = np.linspace(x_min, x_max, n_seg + 1)
            
            for ax in [ax1, ax2, ax3]:
                for boundary in segment_boundaries[1:-1]:  # 跳过首尾边界
                    ax.axvline(boundary, color='k', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
    
    def _plot_error_2d(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """2D误差分布可视化"""
        fig = self._create_figure(figsize=(15, 5))
        
        # 计算误差
        x_global = np.vstack(data['x_segments'])
        u_global = np.vstack(data['u_segments'])
        error = np.abs(prediction - u_global)
        relative_error = error / (np.abs(u_global) + 1e-12)
        
        # 创建网格
        x = np.unique(x_global[:, 0])
        y = np.unique(x_global[:, 1])
        X, Y = np.meshgrid(x, y)
        
        # 重塑误差
        error_grid = error.reshape(len(y), len(x))
        rel_error_grid = relative_error.reshape(len(y), len(x))
        
        # 绝对误差
        ax1 = fig.add_subplot(131)
        im1 = ax1.pcolormesh(X, Y, error_grid, cmap='hot')
        plt.colorbar(im1, ax=ax1, label='Absolute Error')
        ax1.set_title('Absolute Error Distribution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        
        # 相对误差
        ax2 = fig.add_subplot(132)
        im2 = ax2.pcolormesh(X, Y, rel_error_grid, cmap='plasma')
        plt.colorbar(im2, ax=ax2, label='Relative Error')
        ax2.set_title('Relative Error Distribution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        
        # 对数误差
        ax3 = fig.add_subplot(133)
        log_error_grid = np.log10(error_grid + 1e-16)
        im3 = ax3.pcolormesh(X, Y, log_error_grid, cmap='viridis')
        plt.colorbar(im3, ax=ax3, label='log10(Absolute Error)')
        ax3.set_title('Log Error Distribution')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        
        # 添加段边界
        self._add_segment_boundaries_2d([ax1, ax2, ax3])
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
        
    def plot_evolution_step(self, T: float, u: np.ndarray, x: np.ndarray, save_path: Optional[str] = None) -> None:
        """绘制时间演化过程中的单个时间步 - 根据维度自动选择可视化方式"""
        if self.is_1d:
            self._plot_evolution_step_1d(T, u, x, save_path)
        else:
            self._plot_evolution_step_2d(T, u, x, save_path)
    
    def _plot_evolution_step_1d(self, T: float, u: np.ndarray, x: np.ndarray, save_path: Optional[str] = None) -> None:
        """1D时间演化步骤可视化"""
        self.ax.clear()
        
        # 排序以便正确绘制线图
        sort_idx = np.argsort(x[:, 0])
        x_sorted = x[sort_idx, 0]
        u_sorted = u[sort_idx, 0] if u.ndim > 1 else u[sort_idx]
        
        # 绘制解曲线
        self.ax.plot(x_sorted, u_sorted, 'b-', linewidth=3, label=f'u(x, t={T:.3f})')
        self.ax.set_xlabel("x", fontsize=12)
        self.ax.set_ylabel("u", fontsize=12)
        self.ax.set_title(f"Solution at T = {T:.4f}", fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # 设置坐标轴范围
        if hasattr(self.config, 'x_domain'):
            self.ax.set_xlim([self.config.x_domain[0][0], self.config.x_domain[0][1]])
        
        # 动态调整y轴范围
        u_min, u_max = u_sorted.min(), u_sorted.max()
        margin = (u_max - u_min) * 0.1 if u_max != u_min else 0.1
        self.ax.set_ylim([u_min - margin, u_max + margin])
        
        # 添加段边界标记
        if hasattr(self.config, 'n_segments') and hasattr(self.config, 'x_domain'):
            x_min_domain, x_max_domain = self.config.x_domain[0]
            n_seg = self.config.n_segments[0] if isinstance(self.config.n_segments, (list, tuple)) else self.config.n_segments
            segment_boundaries = np.linspace(x_min_domain, x_max_domain, n_seg + 1)
            
            for boundary in segment_boundaries[1:-1]:  # 跳过首尾边界
                self.ax.axvline(boundary, color='red', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.draw()
        plt.pause(0.01)
    
    def _plot_evolution_step_2d(self, T: float, u: np.ndarray, x: np.ndarray, save_path: Optional[str] = None) -> None:
        """2D时间演化步骤可视化"""
        self.ax.clear()
        
        # 绘制3D散点图
        u_vals = u[:, 0] if u.ndim > 1 else u
        scatter = self.ax.scatter(x[:, 0], x[:, 1], u_vals, c=u_vals, cmap="RdBu", s=15, alpha=0.8)
        
        # 设置视角和标签
        self.ax.view_init(elev=30, azim=45)
        self.ax.set_xlabel("x", fontsize=12)
        self.ax.set_ylabel("y", fontsize=12)
        self.ax.set_zlabel("u", fontsize=12)
        self.ax.set_title(f"Solution at T = {T:.4f}", fontsize=14, fontweight='bold')
        
        # 设置坐标轴范围
        if hasattr(self.config, 'x_domain'):
            self.ax.set_xlim([self.config.x_domain[0][0], self.config.x_domain[0][1]])
            self.ax.set_ylim([self.config.x_domain[1][0], self.config.x_domain[1][1]])
            
        # 动态调整z轴范围
        zmin, zmax = u_vals.min(), u_vals.max()
        margin = (zmax - zmin) * 0.1 if zmax != zmin else 0.1
        self.ax.set_zlim([zmin - margin, zmax + margin])
        
        # 添加颜色条
        if not hasattr(self, '_colorbar_2d') or self._colorbar_2d is None:
            self._colorbar_2d = plt.colorbar(scatter, ax=self.ax, shrink=0.5, aspect=20)
            self._colorbar_2d.set_label('u', fontsize=12)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.draw()
        plt.pause(0.01)
    
    def plot_evolution_step_with_reference(self, T: float, u: np.ndarray, x: np.ndarray, 
                                         u_ref: Optional[np.ndarray] = None, save_path: Optional[str] = None) -> None:
        """绘制时间演化步骤与参考解对比"""
        if self.is_1d:
            self._plot_evolution_step_with_reference_1d(T, u, x, u_ref, save_path)
        else:
            # 2D情况暂时使用原有方法
            self._plot_evolution_step_2d(T, u, x, save_path)
    
    def _plot_evolution_step_with_reference_1d(self, T: float, u: np.ndarray, x: np.ndarray, 
                                             u_ref: Optional[np.ndarray] = None, save_path: Optional[str] = None) -> None:
        """1D时间演化步骤与参考解对比可视化"""
        self.ax.clear()
        
        # 排序以便正确绘制线图
        sort_idx = np.argsort(x[:, 0])
        x_sorted = x[sort_idx, 0]
        u_sorted = u[sort_idx, 0] if u.ndim > 1 else u[sort_idx]
        
        # 绘制数值解
        self.ax.plot(x_sorted, u_sorted, 'r-', linewidth=3, label=f'数值解 (T={T:.4f})', alpha=0.8)
        
        # 绘制参考解（如果有）
        if u_ref is not None:
            # 假设参考解的x坐标与数值解相同或可以插值
            if len(u_ref) == len(x_sorted):
                self.ax.plot(x_sorted, u_ref[sort_idx], 'b--', linewidth=2, label='参考解', alpha=0.8)
            else:
                # 如果参考解的网格不同，需要插值到数值解网格上
                from scipy.interpolate import interp1d
                x_ref_range = np.linspace(x_sorted.min(), x_sorted.max(), len(u_ref))
                interp_func = interp1d(x_ref_range, u_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
                u_ref_interp = interp_func(x_sorted)
                self.ax.plot(x_sorted, u_ref_interp, 'b--', linewidth=2, label='参考解', alpha=0.8)
                
                # 计算并显示误差
                error = np.abs(u_sorted - u_ref_interp)
                max_error = np.max(error)
                l2_error = np.sqrt(np.mean(error**2))
                self.ax.text(0.02, 0.98, f'最大误差: {max_error:.2e}\nL2误差: {l2_error:.2e}', 
                           transform=self.ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.ax.set_xlabel("x", fontsize=12)
        self.ax.set_ylabel("u", fontsize=12)
        self.ax.set_title(f"Solution Evolution at T = {T:.4f}", fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # 设置坐标轴范围
        if hasattr(self.config, 'x_domain'):
            self.ax.set_xlim([self.config.x_domain[0][0], self.config.x_domain[0][1]])
        
        # 添加段边界标记
        if hasattr(self.config, 'n_segments') and hasattr(self.config, 'x_domain'):
            x_min_domain, x_max_domain = self.config.x_domain[0]
            n_seg = self.config.n_segments[0] if isinstance(self.config.n_segments, (list, tuple)) else self.config.n_segments
            segment_boundaries = np.linspace(x_min_domain, x_max_domain, n_seg + 1)
            
            for boundary in segment_boundaries[1:-1]:  # 跳过首尾边界
                self.ax.axvline(boundary, color='gray', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.draw()
        plt.pause(0.01)
        
    def close_evolution_plot(self) -> None:
        """关闭时间演化绘图"""
        plt.ioff()
        plt.close()
        
    def plot_mesh_results(
        self,
        data: Dict,
        model: torch.nn.Module,
        preds: np.ndarray,
        title: Optional[str] = None,
        save_gif: bool = False,
        save_png: bool = False,
        gif_filename: str = "animation.gif",
        png_filename: Optional[str] = None,
        fast_mode: bool = False,
        contour_levels: int = 50,
    ) -> None:
        """可视化u分量的流场分布"""
        self._save_gif = save_gif
        
        # 提取配置参数
        ns1, ns2 = self.config.n_segments
        np1, np2 = self.config.points_domain_test
        
        # 创建网格
        x1 = np.linspace(self.config.x_domain[0, 0], self.config.x_domain[0, 1], np1 * ns1)
        x2 = np.linspace(self.config.x_domain[1, 0], self.config.x_domain[1, 1], np2 * ns2)
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        
        # 重组预测值到网格形状
        u_pred = preds[:, 0].reshape(np2 * ns2, np1 * ns1)
        
        # 创建或更新图表
        if not hasattr(self, '_mesh_fig'):
            self._mesh_fig = plt.figure(figsize=(10, 8))
            self._mesh_axes = self._mesh_fig.add_subplot(111)
        
        # 清除当前图表
        self._mesh_axes.clear()
        
        # 绘制等值线图
        contf = self._mesh_axes.contourf(
            x1_grid, 
            x2_grid, 
            u_pred,
            levels=contour_levels,
            cmap='RdBu',
        )
        
        # 设置标题和标签
        if title:
            self._mesh_axes.set_title(title)
        self._mesh_axes.set_xlabel('x')
        self._mesh_axes.set_ylabel('y')
        self._mesh_axes.set_aspect('equal')
        
        # 更新或添加颜色条
        if hasattr(self._mesh_axes, 'colorbar'):
            self._mesh_axes.colorbar.remove()
        self._mesh_axes.colorbar = plt.colorbar(contf, ax=self._mesh_axes)
        self._mesh_axes.colorbar.set_label('u')
        
        # 调整布局
        plt.tight_layout()
        
        # 更新图表
        self._mesh_fig.canvas.draw()
        
        # 保存当前帧为PNG（如果需要）
        if save_png and png_filename:
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        
        # 捕获当前帧用于GIF
        if self._save_gif:
            buf = io.BytesIO()
            self._mesh_fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            self._frames.append(Image.open(buf))
        
        self._mesh_fig.canvas.flush_events()
        plt.pause(0.01)
        
    def plot_mesh_results_final(
        self,
        data: Dict,
        model: torch.nn.Module,
        preds: np.ndarray,
    ) -> None:
        """绘制最终结果：u、v、p分量图和黑白流线图"""
        # 提取配置参数
        ne = self.config.n_eqs
        ns1, ns2 = self.config.n_segments
        np1, np2 = self.config.points_per_segment_test
        
        # 计算总点数和区段
        points_per_section = np1 * np2
        total_sections = ns1 * ns2
        
        # 初始化网格数组
        x1_grid = np.zeros((np1 * ns1, np2 * ns2))
        x2_grid = np.zeros((np1 * ns1, np2 * ns2))
        
        # 填充网格数组
        for section in range(total_sections):
            section_x = section // ns2
            section_y = section % ns2
            
            for point in range(points_per_section):
                point_x = point // np2
                point_y = point % np2
                
                # 获取网格点
                grid_points = data["x_segments"][section][point]
                x1_grid[point_y + section_y * np1, point_x + section_x * np2] = grid_points[0]
                x2_grid[point_y + section_y * np1, point_x + section_x * np2] = grid_points[1]
        
        # 获取唯一坐标以确保网格规则性
        x1_unique = np.unique(x1_grid)
        x2_unique = np.unique(x2_grid)
        
        # 初始化预测数组
        u_pred = np.zeros((np1 * ns1, np2 * ns2, ne))
        
        # 填充预测数组
        for section in range(total_sections):
            for point in range(points_per_section):
                for eq in range(ne):
                    section_x = section // ns2
                    section_y = section % ns2
                    point_x = point // np2
                    point_y = point % np2
                    u_pred[point_y + section_y * np1, point_x + section_x * np2, eq] = (
                        preds[points_per_section * section + point, eq]
                    )
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        
        # 1. U分量图
        fig = self._create_figure(figsize=(8, 6))
        contf_u = plt.contourf(x1_grid, x2_grid, u_pred[:, :, 0], levels=20, cmap='viridis')
        plt.colorbar(contf_u, label='U Velocity')
        plt.title('U Component')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('results/u_component.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. V分量图
        fig = self._create_figure(figsize=(8, 6))
        contf_v = plt.contourf(x1_grid, x2_grid, u_pred[:, :, 1], levels=20, cmap='viridis')
        plt.colorbar(contf_v, label='V Velocity')
        plt.title('V Component')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('results/v_component.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. P分量图
        fig = self._create_figure(figsize=(8, 6))
        if ne >= 3:
            contf_p = plt.contourf(x1_grid, x2_grid, u_pred[:, :, 2], levels=20, cmap='viridis')
            plt.colorbar(contf_p, label='Pressure')
            plt.title('P Component')
        else:
            plt.title('Missing P Component Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('results/p_component.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 黑白流线图
        fig = self._create_figure(figsize=(8, 6))
        u_component = u_pred[:, :, 0].T
        v_component = u_pred[:, :, 1].T
        
        plt.streamplot(
            x1_unique, 
            x2_unique, 
            u_component, 
            v_component, 
            color='k',
            linewidth=1.0,
            density=3.0,
            arrowsize=0,
        )
        plt.title('Streamlines')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('results/streamlines.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 输出Tecplot格式文件
        with open('results/flow_field.dat', 'w') as f:
            f.write('TITLE = "Flow Field Data"\n')
            f.write('VARIABLES = "X", "Y", "U", "V"')
            if ne >= 3:
                f.write(', "P"')
            f.write('\n')
            
            f.write(f'ZONE I={np1 * ns1}, J={np2 * ns2}, F=POINT\n')
            
            for j in range(np2 * ns2):
                for i in range(np1 * ns1):
                    line = f"{x1_grid[i,j]:.6f} {x2_grid[i,j]:.6f} {u_pred[i,j,0]:.6f} {u_pred[i,j,1]:.6f}"
                    if ne >= 3:
                        line += f" {u_pred[i,j,2]:.6f}"
                    f.write(line + '\n')
        
        print("4 separate images and Tecplot format data file have been saved to the results folder")
        
    def update_animation_data(self, it: int, T: float, data_test: Dict, model: torch.nn.Module, 
                            coeffs: np.ndarray, fitter, solver=None) -> None:
        """更新动画数据 - 从求解器移动到可视化器，支持参考解对比"""
        animation_skip = getattr(self.config, "animation_skip", 10)
        
        # 检查是否需要更新动画数据
        if it % animation_skip == 0 or T >= getattr(self.config, 'T', float('inf')):
            # 更新时间历史
            if not hasattr(self, 'time_history_viz'):
                self.time_history_viz = []
                self.solution_history_viz = []
                
            self.time_history_viz.append(T)
            
            # 获取当前时间步的解
            try:
                U_test, _ = fitter.construct(data_test, model, coeffs)
                self.solution_history_viz.append(U_test.copy())
                
                # 可选择实时绘制时间演化步骤
                if getattr(self.config, 'realtime_visualization', False):
                    x_coords = data_test.get('x', data_test.get('x_segments', []))
                    if hasattr(x_coords, '__iter__') and len(x_coords) > 0:
                        if isinstance(x_coords, list):
                            x_plot = np.vstack(x_coords) if x_coords else np.array([[0]])
                        else:
                            x_plot = x_coords
                        
                        # 获取参考解（如果有）
                        u_ref = None
                        if solver and hasattr(solver, 'get_reference_solution_at_time'):
                            u_ref = solver.get_reference_solution_at_time(T)
                            
                        self.plot_evolution_step_with_reference(T, U_test, x_plot, u_ref)
                        
                print(f"  动画数据已更新: T = {T:.6f}, 总帧数 = {len(self.solution_history_viz)}")
                
            except Exception as e:
                print(f"  警告: 动画数据更新失败: {e}")
    
    def get_animation_data(self):
        """获取动画数据"""
        if hasattr(self, 'time_history_viz') and hasattr(self, 'solution_history_viz'):
            return self.time_history_viz, self.solution_history_viz
        else:
            return [], []
    
    def save_animation(self, filename: str = "animation.gif", duration: int = 200) -> None:
        """将捕获的帧保存为GIF动画"""
        if len(self._frames) > 0:
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
            
            self._frames[0].save(
                filename,
                save_all=True,
                append_images=self._frames[1:],
                optimize=True,
                duration=duration,
                loop=0,
            )
            print(f"GIF动画已保存至 {filename}")
            
            self._frames = []
        else:
            print("没有帧可保存")
    
    def create_time_evolution_gif(self, time_history: List[float], solution_history: List[np.ndarray], 
                                data_test: Dict, filename: str = "time_evolution.gif", solver=None) -> None:
        """创建时间演化GIF动画，支持参考解对比"""
        if not time_history or not solution_history:
            print("警告: 没有时间演化数据可创建动画")
            return
            
        try:
            print(f"正在创建时间演化动画: {filename}")
            
            # 获取x坐标
            if 'x_segments' in data_test:
                x_coords = np.vstack(data_test['x_segments'])
            elif 'x' in data_test:
                x_coords = data_test['x']
            else:
                print("警告: 无法获取x坐标数据")
                return
            
            # 清空之前的帧
            self._frames = []
            
            # 为每个时间步生成帧
            for i, (T, solution) in enumerate(zip(time_history, solution_history)):
                # 创建临时图形
                temp_fig = plt.figure(figsize=(10, 6) if self.is_1d else (10, 8))
                
                if self.is_1d:
                    # 1D情况
                    sort_idx = np.argsort(x_coords[:, 0])
                    x_sorted = x_coords[sort_idx, 0]
                    u_sorted = solution.flatten()[sort_idx]
                    
                    ax = temp_fig.add_subplot(111)
                    ax.plot(x_sorted, u_sorted, 'b-', linewidth=3, label='Numerical Solution')
                    
                    # 添加参考解对比（如果有solver且有参考解）
                    if solver and hasattr(solver, 'get_reference_solution_at_time'):
                        u_ref = solver.get_reference_solution_at_time(T)
                        if u_ref is not None:
                            # 插值参考解到数值解的网格上
                            if hasattr(solver, 'reference_solution') and solver.reference_solution is not None:
                                try:
                                    x_ref = solver.reference_solution['x_ref']
                                    if len(u_ref) != len(x_sorted):
                                        from scipy.interpolate import interp1d
                                        interp_func = interp1d(x_ref, u_ref, kind='cubic', 
                                                             bounds_error=False, fill_value='extrapolate')
                                        u_ref_interp = interp_func(x_sorted)
                                    else:
                                        u_ref_interp = u_ref
                                        
                                    ax.plot(x_sorted, u_ref_interp, 'r--', linewidth=2, alpha=0.8, label='Reference Solution')
                                    
                                    # 计算并显示误差
                                    error = np.abs(u_sorted - u_ref_interp)
                                    max_error = np.max(error)
                                    l2_error = np.sqrt(np.mean(error**2))
                                    ax.text(0.02, 0.98, f'Max Error: {max_error:.2e}\\nL2 Error: {l2_error:.2e}', 
                                           transform=ax.transAxes, verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                           fontsize=10)
                                except Exception as e:
                                    print(f"参考解插值失败: {e}")
                    
                    ax.set_xlabel('x', fontsize=12)
                    ax.set_ylabel('u', fontsize=12)
                    ax.set_title(f'T = {T:.4f}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right')
                    
                    # 设置固定的y轴范围以便动画稳定
                    all_solutions = np.concatenate([s.flatten() for s in solution_history])
                    y_min, y_max = all_solutions.min(), all_solutions.max()
                    
                    # 如果有参考解，也考虑其范围
                    if solver and hasattr(solver, 'reference_solution') and solver.reference_solution is not None:
                        try:
                            u_ref_all = solver.reference_solution['u_ref']
                            y_min = min(y_min, u_ref_all.min())
                            y_max = max(y_max, u_ref_all.max())
                        except:
                            pass
                    
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim([y_min - margin, y_max + margin])
                    
                else:
                    # 2D情况
                    ax = temp_fig.add_subplot(111, projection='3d')
                    u_vals = solution.flatten()
                    ax.scatter(x_coords[:, 0], x_coords[:, 1], u_vals, c=u_vals, cmap='RdBu', s=10)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y') 
                    ax.set_zlabel('u')
                    ax.set_title(f'T = {T:.4f}')
                
                plt.tight_layout()
                
                # 将当前帧添加到动画
                buf = io.BytesIO()
                temp_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                self._frames.append(Image.open(buf))
                plt.close(temp_fig)
                
                if (i + 1) % 10 == 0:  # 每10帧打印进度
                    print(f"  已生成 {i + 1}/{len(time_history)} 帧")
            
            # 保存GIF动画
            self.save_animation(filename, duration=200)
            print(f"时间演化动画创建完成: {filename}")
            
        except Exception as e:
            print(f"动画创建失败: {e}")
            
    def _add_segment_boundaries_2d(self, axes_list: List) -> None:
        """为2D图添加段边界 - 辅助方法"""
        if hasattr(self.config, 'x_min') and hasattr(self.config, 'x_max'):
            for n in range(getattr(self, 'Ns', 0)):
                try:
                    x_min = self.config.x_min[n]
                    x_max = self.config.x_max[n]
                    
                    # 绘制段边界
                    for ax in axes_list:
                        ax.plot([x_min[0], x_max[0]], [x_min[1], x_min[1]], 'k--', alpha=0.4, linewidth=1)
                        ax.plot([x_min[0], x_max[0]], [x_max[1], x_max[1]], 'k--', alpha=0.4, linewidth=1)
                        ax.plot([x_min[0], x_min[0]], [x_min[1], x_max[1]], 'k--', alpha=0.4, linewidth=1)
                        ax.plot([x_max[0], x_max[0]], [x_min[1], x_max[1]], 'k--', alpha=0.4, linewidth=1)
                except (IndexError, AttributeError):
                    # Skip boundary drawing if segment boundaries not available
                    pass
    
    def plot_loss_history(self, loss_history: List[float], save_path: Optional[str] = None) -> None:
        """绘制训练损失历史"""
        fig = self._create_figure(figsize=(10, 6))
        
        iterations = range(1, len(loss_history) + 1)
        ax = fig.add_subplot(111)
        ax.semilogy(iterations, loss_history, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('迭代步数', fontsize=12)
        ax.set_ylabel('损失值 (对数尺度)', fontsize=12)
        ax.set_title('训练损失历史', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        if loss_history:
            final_loss = loss_history[-1]
            min_loss = min(loss_history)
            ax.text(0.02, 0.98, f'最终损失: {final_loss:.2e}\n最小损失: {min_loss:.2e}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
    
    def plot_time_evolution_summary(self, time_history: List[float], solution_history: List[np.ndarray], 
                                  save_path: Optional[str] = None) -> None:
        """绘制时间演化总结图"""
        if not time_history or not solution_history:
            print("警告: 没有时间演化数据可绘制")
            return
            
        if self.is_1d:
            self._plot_time_evolution_summary_1d(time_history, solution_history, save_path)
        else:
            self._plot_time_evolution_summary_2d(time_history, solution_history, save_path)
    
    def _plot_time_evolution_summary_1d(self, time_history: List[float], solution_history: List[np.ndarray], 
                                       save_path: Optional[str] = None) -> None:
        """1D时间演化总结图"""
        fig = self._create_figure(figsize=(15, 10))
        
        # 确保数据有效性
        if len(time_history) != len(solution_history):
            print(f"警告: 时间历史长度({len(time_history)})与解历史长度({len(solution_history)})不匹配")
            min_len = min(len(time_history), len(solution_history))
            time_history = time_history[:min_len]
            solution_history = solution_history[:min_len]
        
        if len(time_history) == 0:
            print("警告: 没有时间演化数据")
            return
            
        # 选择几个代表性时间点
        n_snapshots = min(6, len(time_history))
        if len(time_history) == 1:
            snapshot_indices = [0]
        else:
            snapshot_indices = np.linspace(0, len(time_history)-1, n_snapshots, dtype=int)
        
        # 创建子图网格
        for i, idx in enumerate(snapshot_indices):
            ax = fig.add_subplot(2, 3, i+1)
            u = solution_history[idx]
            t = time_history[idx]
            
            # 这里需要x坐标信息，假设从config获取
            if hasattr(self.config, 'x_domain') and hasattr(self.config, 'points_domain_test'):
                n_points = self.config.points_domain_test[0] if isinstance(self.config.points_domain_test, (list, tuple)) else self.config.points_domain_test
                x = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], len(u))
                ax.plot(x, u.flatten(), 'b-', linewidth=2)
            else:
                ax.plot(u.flatten(), 'b-', linewidth=2)
            
            ax.set_title(f'T = {t:.4f}', fontsize=12)
            ax.grid(True, alpha=0.3)
            if i >= 3:  # 下排子图
                ax.set_xlabel('x', fontsize=10)
            if i % 3 == 0:  # 左列子图
                ax.set_ylabel('u', fontsize=10)
        
        plt.suptitle('时间演化快照', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
    
    def _plot_time_evolution_summary_2d(self, time_history: List[float], solution_history: List[np.ndarray], 
                                       save_path: Optional[str] = None) -> None:
        """2D时间演化总结图"""
        fig = self._create_figure(figsize=(15, 10))
        
        # 确保数据有效性
        if len(time_history) != len(solution_history):
            print(f"警告: 时间历史长度({len(time_history)})与解历史长度({len(solution_history)})不匹配")
            min_len = min(len(time_history), len(solution_history))
            time_history = time_history[:min_len]
            solution_history = solution_history[:min_len]
        
        if len(time_history) == 0:
            print("警告: 没有时间演化数据")
            return
        
        # 选择几个代表性时间点
        n_snapshots = min(6, len(time_history))
        if len(time_history) == 1:
            snapshot_indices = [0]
        else:
            snapshot_indices = np.linspace(0, len(time_history)-1, n_snapshots, dtype=int)
        
        for i, idx in enumerate(snapshot_indices):
            ax = fig.add_subplot(2, 3, i+1)
            u = solution_history[idx]
            t = time_history[idx]
            
            # 简单的2D热力图显示
            if u.ndim == 1:
                # 假设是规则网格，需要重塑
                side_length = int(np.sqrt(len(u)))
                if side_length * side_length == len(u):
                    u_grid = u.reshape(side_length, side_length)
                    im = ax.imshow(u_grid, cmap='viridis', aspect='equal')
                    plt.colorbar(im, ax=ax, shrink=0.8)
            
            ax.set_title(f'T = {t:.4f}', fontsize=12)
            ax.set_aspect('equal')
        
        plt.suptitle('时间演化快照', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
    
    def set_visualization_interval(self, interval: int = 1) -> None:
        """设置可视化间隔"""
        self._visualization_interval = max(1, interval)
        print(f"可视化间隔设置为: 每 {self._visualization_interval} 次迭代更新一次")
    
    def reset_animation_data(self):
        """重置动画数据"""
        self.time_history_viz = []
        self.solution_history_viz = []
        self._frames = []
        print("动画数据已重置")
        
    def examin_data(self, data_train: Dict) -> None:
        """检查数据分布"""
        fig = self._create_figure()
        plt.scatter(
            data_train["x_boundary"][:, 0],
            data_train["x_boundary"][:, 1],
            label="边界解",
        )
        plt.scatter(data_train["x"][:, 0], data_train["x"][:, 1], label="精确解")
        plt.legend()
        plt.show()
        
    def examin_net(self, model: torch.nn.Module, data_GPU: torch.Tensor) -> None:
        """检查网络输出并绘制流场"""
        model.eval()
        with torch.no_grad():
            # 创建规则网格
            N = 100
            x1_grid = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], N)
            x2_grid = np.linspace(self.config.x_domain[1][0], self.config.x_domain[1][1], N)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            grid_points = np.column_stack((X1.ravel(), X2.ravel()))
            
            # 转换为张量
            grid_points = torch.tensor(
                grid_points, dtype=torch.float64, device=self.config.device, requires_grad=True
            )
            
            # 获取网络预测
            _, u_pred = model(grid_points)
            
            # 分离速度和压力分量
            u = u_pred[..., 0]
            
            # 转换为numpy数组
            u = u.detach().cpu().numpy()
            
            # 重新调整为网格形状
            U = u.reshape(N, N)
            
            # 创建图表
            fig = self._create_figure(figsize=(15, 5))
            
            # 绘制U分量
            ax1 = fig.add_subplot(111)
            cs1 = ax1.contourf(X1, X2, U, levels=50, cmap="RdBu")
            plt.colorbar(cs1, ax=ax1, label="U速度")
            ax1.set_title("U速度")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            
            plt.tight_layout()
            plt.show() 