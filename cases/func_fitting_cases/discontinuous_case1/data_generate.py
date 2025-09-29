import numpy as np
from typing import Dict

def generate_global_field(x):
    """生成全局场
    
    Args:
        x: 输入坐标数组
        
    Returns:
        np.ndarray: 全局场值
    """
    x_vals = x[:, 0]  # 提取x坐标值
    y_global = np.zeros_like(x_vals)
    
    # 对x<0的部分
    mask_neg = x_vals < 0
    if np.any(mask_neg):
        # 计算sin(kx)的和，k从1到4
        sum_sin = np.zeros_like(x_vals[mask_neg])
        for k in range(1, 5):
            sum_sin += np.sin(k * x_vals[mask_neg])
        y_global[mask_neg] = 5 + sum_sin
    
    # 对x>=0的部分
    mask_pos = x_vals >= 0
    if np.any(mask_pos):
        y_global[mask_pos] = np.cos(10 * x_vals[mask_pos])
    
    # 返回形状正确的数组
    return y_global.reshape(-1, 1)