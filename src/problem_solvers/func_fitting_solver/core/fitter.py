from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver

class FuncFittingFitter(BaseDeepPolyFitter):
    """函数拟合问题的混合拟合器实现"""
    
    def __init__(self, config, data: Dict = None):
        super().__init__(config, data)
        self.data = data
        # 初始化求解器
        self.solver = LinearSolver(verbose=True, use_gpu=True, performance_tracking=True)

    def get_segment_data(self, segment_idx: int) -> Dict:
        """获取指定段的数据"""
        return {
            "x": self.data["x_segments"][segment_idx],
            "u": self.data["u_segments"][segment_idx]
        }

    def build_segment_jacobian(
        self,
        segment_data: Dict[str, Any],
        equations: Dict,
        variables: Dict,
        segment_idx: int,
        config: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建单个段的雅可比矩阵"""
        # 获取数据
        x = segment_data["x"]
        u = segment_data["u"]
        
        eq = []
        for i in range(self.config.n_eqs):
            eq.append(self.equations[f"eq{i}"][segment_idx])
        
        n_points = self.config.n_points
        ne = self.config.n_eqs
        dgN = self.config.dgN

        L = np.zeros((ne, n_points, ne * dgN))
        b = np.zeros((ne, n_points))

        # 获取变量
        U = variables["U"][segment_idx]

        # 构建拟合方程
        L[0] = U
        b[0] = u[:, 0].flatten()

        # 添加空间离散项
        for i in range(ne):
            L[i] += eq[i]

        # 重塑矩阵
        L = np.vstack([L[i] for i in range(ne)])
        b = np.vstack([b[i].reshape(-1, 1) for i in range(ne)])

        return L, b

    def _build_jacobian_nonlinear(
        self, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建雅可比矩阵和残差向量"""
        ns = self.ns
        ne = self.config.n_eqs
        dgN = self.dgN

        # 计算总行数
        total_rows = 0
        rows_per_segment = []
        for i in range(ns):
            n_points = len(self.data["x_segments"][i])
            segment_rows = n_points * ne
            rows_per_segment.append(segment_rows)
            total_rows += segment_rows

        n_cols = ne * dgN * ns

        # 预分配矩阵
        J1 = np.zeros((total_rows, n_cols), dtype=np.float64)
        b1 = np.zeros((total_rows, 1), dtype=np.float64)

        # 处理每个段
        row_start = 0
        for i in range(ns):
            # 计算当前段的索引
            row_end = row_start + rows_per_segment[i]
            col_start = i * ne * dgN
            col_end = (i + 1) * ne * dgN

            # 获取当前段的数据
            x_seg = self.data["x_segments"][i]
            u_seg = self.data["u_segments"][i]

            # 计算段的雅可比和残差
            L, r = self._compute_segment_jacobian_nonlinear(i)

            # 填充矩阵
            J1[row_start:row_end, col_start:col_end] = L
            b1[row_start:row_end] = r

            row_start = row_end

        # 处理约束
        if len(self.A) > 0:
            A_constraints = np.vstack(self.A)
            b_constraints = np.vstack([b_i.reshape(-1, 1) for b_i in self.b])
        else:
            A_constraints = np.zeros((0, n_cols))
            b_constraints = np.zeros((0, 1))

        # 组装最终结果
        J = np.vstack([J1, A_constraints])
        r = np.vstack([b1, b_constraints])

        return J, r

    def _compute_segment_jacobian_nonlinear(
        self,
        segment_idx: int,
        eps: float = 1e-15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算段的雅可比矩阵和残差"""
        eq = []
        for i in range(self.config.n_eqs):
            eq.append(self.equations[f"eq{i}"][segment_idx])

        n_points = len(self.data["x_segments"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.dgN

        L = np.zeros((ne, n_points, ne * dgN))
        b = np.zeros((ne, n_points))

        # 获取变量
        #U = self.variables["U"][segment_idx]

        # 构建拟合方程
        #L[0] = U
        b[0] = self.data["u_segments"][segment_idx][:, 0].flatten()

        # 添加空间离散项
        for i in range(ne):
            L[i] += eq[i]

        # 重塑矩阵
        L = np.vstack([L[i] for i in range(ne)])
        b = np.vstack([b[i].reshape(-1, 1) for i in range(ne)])

        return L, b 