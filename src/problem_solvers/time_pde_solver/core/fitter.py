from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn

from abstract_class.base_fitter import BaseDeepPolyFitter
from algebraic_solver import LinearSolver


class TimePDEFitter(BaseDeepPolyFitter):
    """时间依赖问题的混合拟合器实现"""

    def __init__(self, config, data: Dict = None):
        super().__init__(config, data)

        # 初始化求解器
        self.solver = LinearSolver(
            verbose=True, use_gpu=True, performance_tracking=True
        )

    def get_segment_data(self, data: Dict, segment_idx: int) -> Dict:
        """获取指定段的数据，支持高维索引"""
        return {
            "u_p0": (
                data["u_n_seg"][segment_idx]
                if data["u_n_seg"] is not None and isinstance(data["u_n_seg"], list)
                else data["u_n_seg"]
            ),
            "u_p1": (
                data["u_ng_seg"][segment_idx]
                if data["u_ng_seg"] is not None and isinstance(data["u_ng_seg"], list)
                else data["u_ng_seg"]
            ),
            "f_n": (
                data["f_n_seg"][segment_idx]
                if data["f_n_seg"] is not None and isinstance(data["f_n_seg"], list)
                else data["f_n_seg"]
            ),
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
        # 验证必要参数
        required_params = ["step", "dt"]
        for param in required_params:
            if param not in segment_data:
                raise ValueError(f"Missing required parameter: {param}")

        step = segment_data["step"]
        dt = segment_data["dt"]

        eq = []
        for i in range(config["n_eqs"]):
            eq.append(equations[f"eq{i}"][segment_idx])

        n_points = config["n_points"]
        ne = config["n_eqs"]
        dgN = config["dgN"]

        L = np.zeros((ne, n_points, ne * dgN))
        b = np.zeros((ne, n_points))

        # 获取变量
        U = variables["U"][segment_idx]

        # 根据步骤选择不同的格式
        if step == "1st_order":
            u_p0 = segment_data["u_p0"]
            L[0] = U
            b[0] = u_p0[:, 0].flatten()
        elif step == "pre":
            u_p0 = segment_data["u_p0"]
            f_p0 = segment_data["f_n"]
            L[0] = U / 0.5
            b[0] = u_p0[:, 0].flatten() / 0.5 - f_p0[:, 0].flatten() * dt
        elif step == "corr":
            u_p0 = segment_data["u_p0"]
            u_p1 = segment_data["u_p1"]
            f_p0 = segment_data["f_n"]
            gamma = 0.5  # BDF2格式的系数
            L[0] = U / gamma
            b[0] = u_p0[:, 0].flatten() / gamma - f_p0[:, 0].flatten() * dt

        # 添加空间离散项
        for i in range(ne):
            L[i] += eq[i] * dt

        # 重塑矩阵
        L = np.vstack([L[i] for i in range(ne)])
        b = np.vstack([b[i].reshape(-1, 1) for i in range(ne)])

        return L, b

    def _build_jacobian_nonlinear(
        self, data: Dict, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建雅可比矩阵和残差向量，支持BDF1和BDF2格式"""
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

            # 获取当前段的解
            u_p0_seg = (
                data["u_n_seg"][i]
                if data["u_n_seg"] is not None and isinstance(data["u_n_seg"], list)
                else data["u_n_seg"]
            )
            u_p1_seg = (
                data["u_ng_seg"][i]
                if data["u_ng_seg"] is not None and isinstance(data["u_ng_seg"], list)
                else data["u_ng_seg"]
            )
            f_p0_seg = (
                data["f_n_seg"][i]
                if data["f_n_seg"] is not None and isinstance(data["f_n_seg"], list)
                else data["f_n_seg"]
            )

            # 计算段的雅可比和残差
            L, r = self._compute_segment_jacobian_nonlinear(
                kwargs["step"],
                {"u_p0": u_p0_seg, "u_p1": u_p1_seg, "f_n": f_p0_seg},
                kwargs["dt"],
                i,
            )

            # 填充矩阵
            J1[row_start:row_end, col_start:col_end] = L
            b1[row_start:row_end] = r

            row_start = row_end

        # 处理约束
        A_constraints = np.vstack(self.A)
        b_constraints = np.vstack([b_i.reshape(-1, 1) for b_i in self.b])

        # 组装最终结果
        J = np.vstack([J1, A_constraints])
        r = np.vstack([b1, b_constraints])

        return J, r

    def _compute_segment_jacobian_nonlinear(
        self,
        step: str,
        data: Dict,
        dt: float,
        segment_idx: int,
        eps: float = 1e-15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算段的雅可比矩阵和残差，支持BDF1和BDF2格式"""
        eq = []
        for i in range(self.config.n_eqs):
            eq.append(self.equations[f"eq{i}"][segment_idx])

        n_points = len(self.data["x_segments"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.dgN
        gamma = self.config.gamma_bdf2
        theta = self.config.theta_bdf2

        L = np.zeros((ne, n_points, ne * dgN))
        b = np.zeros((ne, n_points))

        # 获取变量
        U = self.variables["U"][segment_idx]

        # 根据是否有前一时间层的解来选择格式
        if step == "1st_order":
            u_p0 = data["u_p0"]
            L[0] = U
            b[0] = u_p0[:, 0].flatten()
        elif step == "pre":
            u_p0 = data["u_p0"]
            u_p1 = data["u_p1"]
            f_p0 = data["f_n"]
            # BDF2格式：(3u^{n+1} - 4u^n + u^{n-1})/(2dt)
            L[0] = U / 0.5  # 系数3/2
            b[0] = u_p0[:, 0].flatten() / 0.5 - f_p0[:, 0].flatten() * dt
        elif step == "corr":
            u_p0 = data["u_p0"]
            u_p1 = data["u_p1"]
            f_p0 = data["f_n"]
            gamma = 0.5  # BDF2格式的系数
            L[0] = U / gamma
            b[0] = u_p0[:, 0].flatten() / gamma - f_p0[:, 0].flatten() * dt

        # 添加空间离散项
        for i in range(ne):
            L[i] += eq[i] * dt

        # 重塑矩阵
        L = np.vstack([L[i] for i in range(ne)])
        b = np.vstack([b[i].reshape(-1, 1) for i in range(ne)])

        return L, b
