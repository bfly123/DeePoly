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
    """时间相关PDE问题的可视化器"""
    
    def __init__(self, config):
        super().__init__(config)
        # 初始化交互式绘图
        plt.ion()
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._frames = []  # 用于存储动画帧
        self._save_gif = False
        self._visualization_interval = 1
        self._iteration_counter = 0
        self.cbar = None
        
    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """绘制解"""
        fig = self._create_figure()
        
        # 获取全局解
        x_global = np.vstack(data['x_segments'])
        u_global = np.vstack(data['u_segments'])
        pred_global = prediction
        
        # 检查维度并创建相应的网格
        if x_global.shape[1] == 1:
            # 1D情况
            x = np.unique(x_global[:, 0])
            X = x
            Y = None
        else:
            # 2D情况
            x = np.unique(x_global[:, 0])
            y = np.unique(x_global[:, 1])
            X, Y = np.meshgrid(x, y)
        
        # 重塑预测结果
        if Y is not None:
            # 2D情况
            pred_grid = pred_global.reshape(len(y), len(x))
            u_grid = u_global.reshape(len(y), len(x))
        else:
            # 1D情况
            pred_grid = pred_global.flatten()
            u_grid = u_global.flatten()
        
        # 绘制真实解和预测解
        ax1 = fig.add_subplot(121)
        if Y is not None:
            # 2D绘图
            im1 = ax1.pcolormesh(X, Y, u_grid, cmap='viridis')
            plt.colorbar(im1, ax=ax1)
        else:
            # 1D绘图
            ax1.plot(X, u_grid, 'b-', label='真实解')
            ax1.legend()
        ax1.set_title('真实解')
        
        ax2 = fig.add_subplot(122)
        if Y is not None:
            # 2D绘图
            im2 = ax2.pcolormesh(X, Y, pred_grid, cmap='viridis')
            plt.colorbar(im2, ax=ax2)
        else:
            # 1D绘图
            ax2.plot(X, pred_grid, 'r-', label='预测解')
            ax2.legend()
        ax2.set_title('预测解')
        
        # 添加段边界(仅2D情况)
        if Y is not None and hasattr(self.config, 'x_min') and hasattr(self.config, 'x_max'):
            for n in range(self.Ns):
                try:
                    x_min = self.config.x_min[n]
                    x_max = self.config.x_max[n]
                    
                    # 绘制段边界
                    for ax in [ax1, ax2]:
                        ax.plot([x_min[0], x_max[0]], [x_min[1], x_min[1]], 'k--', alpha=0.5)
                        ax.plot([x_min[0], x_max[0]], [x_max[1], x_max[1]], 'k--', alpha=0.5)
                        ax.plot([x_min[0], x_min[0]], [x_min[1], x_max[1]], 'k--', alpha=0.5)
                        ax.plot([x_max[0], x_max[0]], [x_min[1], x_max[1]], 'k--', alpha=0.5)
                except (IndexError, AttributeError):
                    # Skip boundary drawing if segment boundaries not available
                    pass
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
        
    def plot_error(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """绘制误差分布"""
        fig = self._create_figure()
        
        # 计算误差
        x_global = np.vstack(data['x_segments'])
        u_global = np.vstack(data['u_segments'])
        error = np.abs(prediction - u_global)
        
        # 检查维度并创建网格
        if x_global.shape[1] == 1:
            # 1D情况 - 使用线图显示误差
            x = x_global[:, 0]
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            error_sorted = error.flatten()[sort_idx]
            
            ax = fig.add_subplot(111)
            ax.plot(x_sorted, error_sorted, 'r-', linewidth=2, label='误差')
            ax.set_title('误差分布')
            ax.set_xlabel('x')
            ax.set_ylabel('|误差|')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # 2D情况 - 使用原有的颜色图
            x = np.unique(x_global[:, 0])
            y = np.unique(x_global[:, 1])
            X, Y = np.meshgrid(x, y)
            
            # 重塑误差
            error_grid = error.reshape(len(y), len(x))
            
            # 绘制误差分布
            ax = fig.add_subplot(111)
            im = ax.pcolormesh(X, Y, error_grid, cmap='hot')
            plt.colorbar(im, ax=ax)
            ax.set_title('误差分布')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # 添加段边界(仅2D情况)
            if hasattr(self.config, 'x_min') and hasattr(self.config, 'x_max'):
                for n in range(self.Ns):
                    try:
                        x_min = self.config.x_min[n]
                        x_max = self.config.x_max[n]
                        
                        # 绘制段边界
                        ax.plot([x_min[0], x_max[0]], [x_min[1], x_min[1]], 'k--', alpha=0.5)
                        ax.plot([x_min[0], x_max[0]], [x_max[1], x_max[1]], 'k--', alpha=0.5)
                        ax.plot([x_min[0], x_min[0]], [x_min[1], x_max[1]], 'k--', alpha=0.5)
                        ax.plot([x_max[0], x_max[0]], [x_min[1], x_max[1]], 'k--', alpha=0.5)
                    except (IndexError, AttributeError):
                        # Skip boundary drawing if segment boundaries not available
                        pass
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
        
    def plot_evolution_step(self, T: float, u: np.ndarray, x: np.ndarray, save_path: Optional[str] = None) -> None:
        """绘制时间演化过程中的单个时间步"""
        self.ax.clear()
        
        # 检查维度并选择合适的绘图方式
        if x.shape[1] == 1:
            # 1D情况 - 使用线图
            # 重新初始化为2D图
            if hasattr(self.ax, 'zaxis'):
                plt.close(self.fig)
                self.fig = plt.figure(figsize=(10, 6))
                self.ax = self.fig.add_subplot(111)
            
            # 排序以便正确绘制线图
            sort_idx = np.argsort(x[:, 0])
            x_sorted = x[sort_idx, 0]
            u_sorted = u[sort_idx, 0] if u.ndim > 1 else u[sort_idx]
            
            self.ax.plot(x_sorted, u_sorted, 'b-', linewidth=2)
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("u")
            self.ax.set_title(f"T = {T:.3f}")
            self.ax.grid(True, alpha=0.3)
            
            # 设置坐标轴范围
            if hasattr(self.config, 'x_domain'):
                self.ax.set_xlim([self.config.x_domain[0][0], self.config.x_domain[0][1]])
        else:
            # 2D情况 - 使用3D散点图
            # 重新初始化为3D图(如果需要)
            if not hasattr(self.ax, 'zaxis'):
                plt.close(self.fig)
                self.fig = plt.figure(figsize=(10, 10))
                self.ax = self.fig.add_subplot(111, projection="3d")
            
            # 绘制3D散点图
            u_vals = u[:, 0] if u.ndim > 1 else u
            self.ax.scatter(x[:, 0], x[:, 1], u_vals, c=u_vals, cmap="RdBu", s=10)
            
            # 设置视角和标签
            self.ax.view_init(elev=30, azim=45)
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("u")
            self.ax.set_title(f"T = {T:.3f}")
            
            # 设置坐标轴范围
            if hasattr(self.config, 'x_domain'):
                self.ax.set_xlim([self.config.x_domain[0][0], self.config.x_domain[0][1]])
                self.ax.set_ylim([self.config.x_domain[1][0], self.config.x_domain[1][1]])
                zmin, zmax = u_vals.min(), u_vals.max()
                margin = (zmax - zmin) * 0.1
                self.ax.set_zlim([zmin - margin, zmax + margin])
        
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
            
    def set_visualization_interval(self, interval: int = 1) -> None:
        """设置可视化间隔"""
        self._visualization_interval = max(1, interval)
        print(f"Visualization interval set to: update every {self._visualization_interval} iterations")
        
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