import numpy as np
from typing import Dict

def generate_global_field( x):
    """生成全局场
    
    Args:
        config: 配置对象
        x_global: 全局点坐标
        mode: 数据模式，"train"或"test"
        
    Returns:
        np.ndarray: 全局场值
    """
    # 计算目标函数值（针对test_sin的正弦函数）
    y_global = np.sin(2 * np.pi * x[:, 0]) + 0.2 * np.exp(1.3 * x[:, 0])
    
    # 如果是训练数据，添加随机噪声
    np.random.seed(1)  # 使用配置中的种子保证可重复性
    noise_level = 0.05  # 噪声水平，可以根据需要调整
    noise = np.random.normal(0, noise_level, x.shape[0])
    y = y_global#+ noise
    
    # 返回形状正确的数组
    return y.reshape(-1, 1)