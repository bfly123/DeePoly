from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver


class LinearPDEFitter(BaseDeepPolyFitter):
    """线性偏微分方程拟合器"""
    
    def __init__(self, config, data: Dict = None):
        super().__init__(config, data)
        self.data = data
        # 初始化求解器
        self.solver = LinearSolver(verbose=True, use_gpu=True, performance_tracking=True)

#    def get_segment_data(self, segment_idx: int) -> Dict:
#        """获取指定段的数据"""
#        return {
#            "x_norm": self.data["x_segments_norm"][segment_idx],
#            "source": self.data["source_segments"][segment_idx],
#            "boundary_segments_dict": self.data["boundary_segments_dict"][segment_idx]
#        }

    def _build_segment_jacobian(
        self,
        segment_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建单个段的雅可比矩阵"""
        # 获取数据
        source = self.data["source_segments"][segment_idx]
        
        eq = []
        for i in range(self.config.n_eqs):
            eq.append(self.equations[f"eq{i}"][segment_idx])
        
        n_points = self.data["x_segments_norm"][segment_idx].shape[0]
        ne = self.n_eqs
        dgN = self.dgN

        L = np.zeros((ne, n_points, ne * dgN))
        b = np.zeros((ne, n_points))

        # 构建拟合方程
        b[0] = source[:].flatten()

        # 添加空间离散项
        for i in range(ne):
            L[i] = eq[i]

        # 重塑矩阵
        L = np.vstack([L[i] for i in range(ne)])
        b = np.vstack([b[i].reshape(-1, 1) for i in range(ne)])

        return L, b