import numpy as np
import torch
from math import pi, sin
# if not in config.json use here
def generate_source_term(x):
  """
  生成泊松方程的源项 f = 2π²sin(πx)sin(πy)
  """
  x_coords = x[:, 0]
  y_coords = x[:, 1]
  f = -np.sin(4*pi* x_coords) * np.sin(4*pi * y_coords)
  return f.reshape(-1, 1)

def generate_reference_solution(x):
  """
  生成泊松方程的精确解 u(x,y) = sin(πx)sin(πy)
  """
  x_coords = x[:, 0]
  y_coords = x[:, 1]
  u = 0.5/(4*pi)**2 * np.sin(4*pi * x_coords) * np.sin(4*pi * y_coords)

  return u.reshape(-1, 1)

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
    u_exact = np.sin(4*pi * x_coords) * np.sin(4*pi * y_coords)
    
    return u_exact.reshape(-1, 1)