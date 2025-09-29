import numpy as np
import torch

def generate_global_field(x):
    # 计算更复杂的二维光滑函数
    def complex_smooth_function(x, y):
        # 多个高斯峰
        gaussian1 = np.exp(-(x**2 + y**2))
        gaussian2 = 0.7 * np.exp(-((x-1.5)**2 + (y-1.0)**2)/1.2)
        gaussian3 = 0.5 * np.exp(-((x+1.0)**2 + (y+1.5)**2)/0.8)
        
        # 正弦波
        sine_term = 0.2 * np.sin(3*x) * np.cos(2*y)
        
        # 多项式项
        poly_term = 0.1 * (x**2 - y**2) + 0.05 * x * y**2
        
        # 组合所有项
        return gaussian1 + gaussian2 + gaussian3 + sine_term + poly_term
    
    u_train = complex_smooth_function(x[:,0], x[:,1]).reshape(-1, 1)
    return u_train