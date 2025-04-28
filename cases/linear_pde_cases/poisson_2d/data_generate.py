import numpy as np
import torch
from math import pi, sin

def generate_source_term(x):
  """
  生成泊松方程的源项 f = 2π²sin(πx)sin(πy)
  """
  x_coords = x[:, 0]
  y_coords = x[:, 1]
  f = 2 * (pi**2) * np.sin(pi * x_coords) * np.sin(pi * y_coords)
  return f.reshape(-1, 1)

def generate_global_field(x):
    """
    生成泊松方程的精确解 u(x,y) = sin(πx)sin(πy)
    
    该函数对应的泊松方程为 -Δu = f，其中 f = 2π²sin(πx)sin(πy)
    
    Args:
        x: 输入坐标 (n_points, 2)
    Returns:
        u: 精确解 (n_points, 1)
    """
    # 提取坐标
    x_coords = x[:, 0]
    y_coords = x[:, 1]
    
    # 计算精确解
    u_exact = np.sin(pi * x_coords) * np.sin(pi * y_coords)
    
    return u_exact.reshape(-1, 1)

def generate_source_term(x):
    """
    生成泊松方程的源项 f = 2π²sin(πx)sin(πy)
    
    Args:
        x: 输入坐标 (n_points, 2)
    Returns:
        f: 源项 (n_points, 1)
    """
    # 提取坐标
    x_coords = x[:, 0]
    y_coords = x[:, 1]
    
    # 计算拉普拉斯算子作用的结果
    # -Δu = -(∂²u/∂x² + ∂²u/∂y²) = 2π²sin(πx)sin(πy)
    f = 2 * (pi**2) * np.sin(pi * x_coords) * np.sin(pi * y_coords)
    
    return f.reshape(-1, 1)

def exact_solution(x, y):
    """
    返回精确解的函数形式，用于可视化
    
    Args:
        x, y: 坐标（可以是标量或数组）
    Returns:
        u: 精确解
    """
    return np.sin(pi * x) * np.sin(pi * y) 