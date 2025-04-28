from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from abstract_class.config.base_visualize import BaseVisualizer

class FuncFittingVisualizer(BaseVisualizer):
    """函数拟合问题的可视化器"""
    
    def plot_solution(
        self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None
    ) -> None:
        """绘制拟合结果
        
        Args:
            data: 包含输入数据的字典
            prediction: 预测结果
            save_path: 保存路径
        """
        # 创建图形
        fig = self._create_figure(figsize=(12, 8))
        
        # 获取数据
        x_segments = data["x_segments"]
        y_segments = data["y_segments"]
        
        # 绘制每个段的结果
        start_idx = 0
        for i in range(self.Ns):
            # 获取当前段的数据
            segment_size = x_segments[i].shape[0]
            end_idx = start_idx + segment_size
            
            # 获取物理坐标
            x_phys = self._normalize_to_physical(x_segments[i], i)
            
            # 绘制真实值
            plt.plot(x_phys[:, 0], y_segments[i][:, 0], 'b.', label='True' if i == 0 else None)
            
            # 绘制预测值
            plt.plot(x_phys[:, 0], prediction[start_idx:end_idx, 0], 'r-', label='Prediction' if i == 0 else None)
            
            start_idx = end_idx
            
        # 设置图形属性
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Function Fitting Results')
        plt.legend()
        plt.grid(True)
        
        # 保存图形
        if save_path:
            self._save_figure(fig, save_path)
            
        # 关闭图形
        self._close_figure(fig) 