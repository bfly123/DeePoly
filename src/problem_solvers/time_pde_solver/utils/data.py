import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import os
import importlib.util
from src.abstract_class.config.base_data import BaseDataGenerator


class TimePDEDataGenerator(BaseDataGenerator):
    """时间相关PDE问题的数据生成器"""

    def __init__(self, config):
        super().__init__(config)
        """Initialize data generator
        
        Args:
            config: Configuration object
        """
        self.boundary_conditions = self.read_boundary_conditions()

    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        """Generate global field values - 初始条件

        Args:
            x_global: Global point coordinates

        Returns:
            np.ndarray: Global field values (initial conditions)
        """
        u_global = np.zeros((x_global.shape[0], self.n_eqs))
        
        # 检查维度
        if x_global.shape[1] == 1:
            # 1D case - 使用config中定义的初始条件
            if self.config.Initial_conditions:
                ic = self.config.Initial_conditions[0]
                import sympy as sp
                x = sp.Symbol('x')
                pi = sp.pi
                expr = sp.sympify(ic['value'])
                func = sp.lambdify(x, expr, 'numpy')
                u_global[:, 0] = func(x_global[:, 0])
            else:
                # 默认初始条件
                u_global[:, 0] = np.cos(np.pi * x_global[:, 0])
        else:
            # 2D case
            x_center = 0.3
            y_center = 0.3
            width = 0.2
            sharpness = 50
            
            x_transition = 0.5 * (np.tanh(sharpness * (x_global[:, 0] - (x_center - width/2))) - 
                                 np.tanh(sharpness * (x_global[:, 0] - (x_center + width/2))))
            y_transition = 0.5 * (np.tanh(sharpness * (x_global[:, 1] - (y_center - width/2))) - 
                                 np.tanh(sharpness * (x_global[:, 1] - (y_center + width/2))))
            
            u_global[:, 0] = x_transition * y_transition + 1
        
        return u_global

    def generate_data(self, mode: str = "train") -> Dict:
        """Generate training or test data

        Args:
            mode: Data mode, "train" or "test"

        Returns:
            data: Data dictionary containing x (coordinates) and U (values)
        """
        # Generate domain points
        x_global = self._generate_global_points(mode)
        initial_field = self.generate_global_field(x_global)  # 初始条件场
        global_boundary_dict = self.read_boundary_conditions()
        x_segments, masks = self.split_global_points(x_global)
        initial_segments = self.split_global_field(masks, initial_field)
        x_segments_norm, x_swap, x_swap_norm, boundary_segments_dict = (
            self._process_segments(x_segments, global_boundary_dict)
        )

        return self._prepare_output_dict(
                x_segments,
                initial_segments,
                global_boundary_dict,
                x_segments_norm,
                x_swap,
                x_swap_norm,
                boundary_segments_dict,
        )

    def _prepare_output_dict(self, *args) -> Dict:
        """Prepare output data dictionary"""
        [
            x_segments,
            initial_segments,
            global_boundary_dict,
            x_segments_norm,
            x_swap,
            x_swap_norm,
            boundary_segments_dict,
        ] = args

        return {
            "x": np.vstack(x_segments),
            "x_segments": x_segments,
            "U": np.vstack(initial_segments),  # 初始条件数据 (兼容旧接口)
            "U_seg": initial_segments,         # 分段初始条件数据 (兼容旧接口)
            "u_segments": initial_segments,    # 可视化兼容
            "initial": np.vstack(initial_segments),
            "initial_segments": initial_segments,
            "global_boundary_dict": global_boundary_dict,
            "x_segments_norm": x_segments_norm,
            "x_swap": x_swap,
            "x_swap_norm": x_swap_norm,
            "boundary_segments_dict": boundary_segments_dict,
        }