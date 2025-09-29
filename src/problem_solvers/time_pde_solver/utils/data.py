import numpy as np
from typing import Dict
from src.abstract_class.config.base_data import BaseDataGenerator

class TimePDEDataGenerator(BaseDataGenerator):
    """时间相关PDE问题的数据生成器"""
    
    def generate_domain_data(self, mode="train"):
        """生成计算域内部的数据点
        
        Args:
            mode: 数据模式，"train"或"test"
        
        Returns:
            dict: 包含生成数据的字典
        """
        # 确定点数量
        if mode == "train":
            nx, ny = self.config.points_domain, self.config.points_domain
        else:
            nx, ny = self.config.points_domain_test
        
        # 生成网格
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # 初始化解场
        u0 = np.zeros((nx*ny, 1))
        
        # 特殊初始条件：中心高斯脉冲
        for i in range(nx):
            for j in range(ny):
                # 计算到中心的距离
                r = np.sqrt((X[i,j]-0.5)**2 + (Y[i,j]-0.5)**2)
                # 高斯脉冲
                u0[i*ny+j] = np.exp(-50*r**2)
        
        # 整理数据
        xy = np.column_stack((X.flatten(), Y.flatten()))
        data = {
            "x": xy,
            "u": u0
        }
        
        return data

    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        """生成全局初始场，使用tanh函数创建平滑的初始分布"""
        u_global = np.zeros((x_global.shape[0], self.n_eqs))
        
        # 定义方形区域的中心和大小
        x_center = 0.3  # x方向中心
        y_center = 0.3  # y方向中心
        width = 0.2     # 区域宽度
        sharpness = 50  # 增加陡峭程度
        
        # 使用tanh创建平滑过渡
        x_transition = 0.5 * (np.tanh(sharpness * (x_global[:, 0] - (x_center - width/2))) - 
                             np.tanh(sharpness * (x_global[:, 0] - (x_center + width/2))))
        y_transition = 0.5 * (np.tanh(sharpness * (x_global[:, 1] - (y_center - width/2))) - 
                             np.tanh(sharpness * (x_global[:, 1] - (y_center + width/2))))
        
        # 组合x和y方向的过渡函数
        u_global[:, 0] = x_transition * y_transition + 1
        
        return u_global

    def generate_data(self, mode: str = "train") -> Dict:
        """生成训练/测试数据"""
        # 1. 生成全局点和场
        x_global = self._generate_global_points(mode)
        u_global = self.generate_global_field(x_global)
        
        # 2. 切分到局部段
        x_segments, masks = self.split_global_points(x_global)
        u_segments = self.split_global_field(masks, u_global)
        
        # 3. 生成交换点和归一化数据
        x_swap, x_swap_norm, x_segments_norm = self._process_segments(x_segments)
        
        # 4. 处理边界条件
        x_boundary, u_boundary, x_boundary_seg, u_boundary_seg = self._process_boundaries(x_swap)
        
        # 5. 准备输出数据
        return self._prepare_output_dict(
            x_segments, u_segments, x_segments_norm,
            x_swap, x_swap_norm,
            x_boundary, u_boundary,
            x_boundary_seg, u_boundary_seg
        )
