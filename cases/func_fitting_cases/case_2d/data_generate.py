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

def generate_source_term(x):
    """Generate source term for linear PDE problems

    Args:
        x: Input coordinates (n_points, 2)

    Returns:
        source: Source term values (n_points, 1)
    """
    # Create a complex source term based on spatial coordinates
    x_coords = x[:, 0]
    y_coords = x[:, 1]

    # Multi-peak source term
    source1 = 2.0 * np.exp(-((x_coords-1.0)**2 + (y_coords-1.0)**2))
    source2 = -1.5 * np.exp(-((x_coords+1.0)**2 + (y_coords+1.0)**2)/0.5)
    source3 = 0.8 * np.sin(2*np.pi*x_coords) * np.cos(2*np.pi*y_coords)

    source = source1 + source2 + source3
    return source.reshape(-1, 1)

def generate_reference_solution(x):
    """Generate reference solution for testing

    Args:
        x: Input coordinates (n_points, 2)

    Returns:
        reference: Reference solution values (n_points, 1)
    """
    # Use the same complex function as generate_global_field for testing
    x_coords = x[:, 0]
    y_coords = x[:, 1]

    # Multi-gaussian reference solution
    gaussian1 = np.exp(-(x_coords**2 + y_coords**2))
    gaussian2 = 0.7 * np.exp(-((x_coords-1.5)**2 + (y_coords-1.0)**2)/1.2)
    gaussian3 = 0.5 * np.exp(-((x_coords+1.0)**2 + (y_coords+1.5)**2)/0.8)

    # Sinusoidal component
    sine_term = 0.2 * np.sin(3*x_coords) * np.cos(2*y_coords)

    # Polynomial component
    poly_term = 0.1 * (x_coords**2 - y_coords**2) + 0.05 * x_coords * y_coords**2

    reference = gaussian1 + gaussian2 + gaussian3 + sine_term + poly_term
    return reference.reshape(-1, 1)