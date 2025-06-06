import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List, Any
import os

class LinearPDEVisualizer:
    """线性偏微分方程可视化工具"""
    
    def __init__(self, config=None):
        """初始化可视化工具
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.n_dim = config.n_dim if config else None
    
    def plot_solution(
        self, data: Dict, prediction: np.ndarray, 
        save_path: Optional[str] = None,
        title: str = "PDE Solution"
    ):
        """绘制PDE求解结果
        
        Args:
            data: 包含输入数据的字典
            prediction: 预测结果
            save_path: 保存路径
            title: 图表标题
        """
        if self.n_dim == 1:
            self._plot_1d_solution(data, prediction, save_path, title)
        elif self.n_dim == 2:
            self._plot_2d_solution(data, prediction, save_path, title)
        else:
            print(f"暂不支持 {self.n_dim} 维结果的可视化")
    
    def _plot_1d_solution(
        self, data: Dict, prediction: np.ndarray, 
        save_path: Optional[str] = None,
        title: str = "1D PDE Solution"
    ):
        """绘制一维PDE解
        
        Args:
            data: 包含输入数据的字典
            prediction: 预测结果
            save_path: 保存路径
            title: 图表标题
        """
        x = data["x"].squeeze()
        u_true = data["u"].squeeze()
        u_pred = prediction.squeeze()
        
        # 按x坐标排序
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        u_true = u_true[sort_idx]
        u_pred = u_pred[sort_idx]
        
        # 计算误差
        abs_error = np.abs(u_pred - u_true)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制解
        ax1.plot(x, u_true, 'b-', label='True Solution', linewidth=2)
        ax1.plot(x, u_pred, 'r--', label='Predicted Solution', linewidth=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True)
        
        # 绘制误差
        ax2.semilogy(x, abs_error, 'k-', linewidth=1.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Absolute Error (log scale)')
        ax2.set_title('Absolute Error')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"图像已保存至 {save_path}")
            
            # 保存误差数据
            error_data_path = os.path.splitext(save_path)[0] + '_error.dat'
            error_data = np.column_stack((x, abs_error))
            np.savetxt(error_data_path, error_data, header='x error', comments='')
            print(f"误差数据已保存至 {error_data_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_2d_solution(
        self, data: Dict, prediction: np.ndarray, 
        save_path: Optional[str] = None,
        title: str = "2D PDE Solution"
    ):
        """绘制二维PDE解
        
        Args:
            data: 包含输入数据的字典
            prediction: 预测结果
            save_path: 保存路径
            title: 图表标题
        """
        x = data["x"]
        u_true = data["u"].squeeze()
        u_pred = prediction.squeeze()
        
        # 提取坐标
        x_coords = x[:, 0]
        y_coords = x[:, 1]
        
        # 计算误差
        abs_error = np.abs(u_pred - u_true)
        
        # 创建网格用于绘图
        n_grid = 100
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        xi = np.linspace(x_min, x_max, n_grid)
        yi = np.linspace(y_min, y_max, n_grid)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # 使用散点插值
        from scipy.interpolate import griddata
        
        Ui_true = griddata((x_coords, y_coords), u_true, (Xi, Yi), method='cubic')
        Ui_pred = griddata((x_coords, y_coords), u_pred, (Xi, Yi), method='cubic')
        Ei = griddata((x_coords, y_coords), abs_error, (Xi, Yi), method='cubic')
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 绘制真实解
        c1 = axes[0, 0].contourf(Xi, Yi, Ui_true, 50, cmap='viridis')
        axes[0, 0].set_title('True Solution')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        fig.colorbar(c1, ax=axes[0, 0])
        
        # 绘制预测解
        c2 = axes[0, 1].contourf(Xi, Yi, Ui_pred, 50, cmap='viridis')
        axes[0, 1].set_title('Predicted Solution')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        fig.colorbar(c2, ax=axes[0, 1])
        
        # 绘制误差（线性刻度）
        c3 = axes[1, 0].contourf(Xi, Yi, Ei, 50, cmap='hot')
        axes[1, 0].set_title('Absolute Error')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        fig.colorbar(c3, ax=axes[1, 0])
        
        # 绘制误差剖面线
        mid_idx = n_grid // 2
        axes[1, 1].semilogy(xi, Ei[mid_idx, :], 'b-', label='Error at y=mid')
        axes[1, 1].semilogy(yi, Ei[:, mid_idx], 'r--', label='Error at x=mid')
        axes[1, 1].set_title('Error Profiles (log scale)')
        axes[1, 1].set_xlabel('Coordinate')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path)
            print(f"图像已保存至 {save_path}")
            
            # 保存误差数据
            error_data_path = os.path.splitext(save_path)[0] + '_error.dat'
            error_data = np.column_stack((x_coords, y_coords, abs_error))
            np.savetxt(error_data_path, error_data, header='x y error', comments='')
            print(f"误差数据已保存至 {error_data_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_convergence(
        self, errors: List[float], 
        save_path: Optional[str] = None,
        title: str = "Convergence History"
    ):
        """绘制收敛曲线
        
        Args:
            errors: 误差列表
            save_path: 保存路径
            title: 图表标题
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制收敛曲线（对数刻度）
        plt.semilogy(range(1, len(errors) + 1), errors, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Error (log scale)')
        plt.title(title)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"收敛曲线已保存至 {save_path}")
        else:
            plt.show()
        
        plt.close() 