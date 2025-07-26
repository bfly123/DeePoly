from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver


class TimePDEFitter(BaseDeepPolyFitter):
    """时间依赖问题的混合拟合器实现 - 基于IMEX-RK(2,2,2)方法"""

    def __init__(self, config, data: Dict = None):
        super().__init__(config, data)

        # 初始化求解器
        self.solver = LinearSolver(
            verbose=True, use_gpu=True, performance_tracking=True
        )

        # IMEX-RK(2,2,2) 参数 (根据公式3.3)
        self.gamma = (2 - np.sqrt(2)) / 2  # ≈ 0.2928932

        # Butcher表系数
        self.A_imp = np.array([[self.gamma, 0], [1 - 2 * self.gamma, self.gamma]])

        self.A_exp = np.array([[0, 0], [1 - self.gamma, 0]])

        self.b = np.array([0.5, 0.5])

        # 存储阶段解
        self.stage_solutions = {}

        # 维护当前的段级解值，避免每次重新计算分段
        self.u_seg_current = None  # List[np.ndarray] - 当前时间步的段级解值
        self.u_seg_prev = None  # List[np.ndarray] - 前一时间步的段级解值

    def imex_rk_time_step(self, u_n: np.ndarray, dt: float) -> np.ndarray:
        """执行IMEX-RK(2,2,2)时间步进 - 基于解值的操作

        根据Doc/Time_Scheme.md 3.3和3.4节实现:
        输入: u_n - 当前时间步的解值
        输出: u_new - 新时间步的解值 u^{n+1}
        """

        # 初始化或更新当前段级解值
        self.u_seg_current = self.global_to_segments(u_n)

        # 阶段1: 求解 U^(1)
        u_seg_stage1, coeffs_stage1 = self._solve_imex_stage_u_seg(
            self.u_seg_current, None, dt, stage=1
        )

        # 阶段2: 求解 U^(2)
        u_seg_stage2, coeffs_stage2 = self._solve_imex_stage_u_seg(
            self.u_seg_current, u_seg_stage1, dt, stage=2
        )

        # 最终更新: 计算 u^{n+1} (根据公式3.4)
        u_seg_new = self._imex_final_update_u_seg(
            self.u_seg_current,
            u_seg_stage1,
            u_seg_stage2,
            dt,
            coeffs_stage1,
            coeffs_stage2,
        )

        # 更新维护的段级解值
        self.u_seg_prev = self.u_seg_current
        self.u_seg_current = u_seg_new

        # 转换回全局数组
        u_new = self.segments_to_global(u_seg_new)

        return u_new

    def _solve_imex_stage_u_seg(
        self,
        u_n_seg: List[np.ndarray],
        u_prev_seg: List[np.ndarray],
        dt: float,
        stage: int,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """求解IMEX-RK阶段 - 直接使用段级解值操作

        Args:
            u_n_seg: 当前时间步的段级解值列表
            u_prev_seg: 前一阶段的段级解值列表
            dt: 时间步长
            stage: 阶段编号 (1 或 2)

        Returns:
            Tuple[List[np.ndarray], np.ndarray]: (阶段解的段级列表, 阶段系数)
        """

        # 准备阶段数据 - 直接传递段级数据
        stage_data = {
            "u_n_seg": u_n_seg,
            "u_prev_seg": u_prev_seg,
            "dt": dt,
            "stage": stage,
            "gamma": self.gamma,
        }

        # 使用base_fitter的fit方法求解阶段系数
        coeffs_stage = self.fit(**stage_data)

        # 将系数转换为解值
        u_stage_global, u_stage_segments = self.construct(
            self.data, self._current_model, coeffs_stage
        )

        return u_stage_segments, coeffs_stage

    def _imex_final_update_u_seg(
        self,
        u_n_seg: List[np.ndarray],
        u_seg_stage1: List[np.ndarray],
        u_seg_stage2: List[np.ndarray],
        dt: float,
        coeffs_stage1: np.ndarray,
        coeffs_stage2: np.ndarray,
    ) -> List[np.ndarray]:
        """IMEX-RK最终更新步骤 - 直接使用段级解值和系数

        根据Doc/Time_Scheme.md 3.4节公式:
        u^{n+1} = u^n + Δt/2 * [L1 β^(1) + L1 β^(2) + N(u^(1)) + N(u^(2)) + L2⊙F(u^(1)) β^(1) + L2⊙F(u^(2)) β^(2)]

        注意: L1和L2算子作用于系数β，N算子作用于解值u
        """

        u_seg_new = []

        for segment_idx in range(self.ns):
            n_points = len(self.data["x_segments_norm"][segment_idx])
            u_n_seg_current = u_n_seg[segment_idx]

            # 计算各阶段的算子贡献
            total_contribution = np.zeros_like(u_n_seg_current)

            for stage_idx, (u_seg_stage, coeffs_stage, weight) in enumerate(
                [
                    (u_seg_stage1[segment_idx], coeffs_stage1, self.b[0]),
                    (u_seg_stage2[segment_idx], coeffs_stage2, self.b[1]),
                ]
            ):
                stage_contribution = np.zeros_like(u_n_seg_current)

                # 提取当前段的系数
                ne = self.config.n_eqs
                dgN = self.dgN

                for eq_idx in range(ne):
                    # 获取当前段当前方程的系数
                    beta_seg_stage = coeffs_stage[segment_idx, eq_idx, :]

                    # L1项: L1 @ β^(i)
                    if self.has_operator("L1"):
                        L1_seg = self._linear_operators[segment_idx].get("L1", None)
                        if L1_seg is not None:
                            stage_contribution += L1_seg @ beta_seg_stage

                # N项: N(u^(i)) - 非线性，直接以u值作为输入
                if self.has_operator("N"):
                    N_vals = self.N_func(self._features[segment_idx], u_seg_stage)
                    if hasattr(N_vals, "__len__") and len(N_vals) == n_points:
                        stage_contribution += N_vals
                    else:
                        stage_contribution += N_vals

                # L2⊙F项: L2 @ (F(u^(i)) * β^(i))
                if self.has_operator("L2") and self.has_operator("F"):
                    L2_seg = self._linear_operators[segment_idx].get("L2", None)
                    if L2_seg is not None:
                        F_vals = self.F_func(self._features[segment_idx], u_seg_stage)

                        for eq_idx in range(ne):
                            beta_seg_stage = coeffs_stage[segment_idx, eq_idx, :]
                            if hasattr(F_vals, "__len__") and len(F_vals) == n_points:
                                stage_contribution += L2_seg @ (F_vals * beta_seg_stage)
                            else:
                                stage_contribution += F_vals * (L2_seg @ beta_seg_stage)

                # 加权累加
                total_contribution += weight * stage_contribution

            # 最终更新: u^{n+1} = u^n + Δt * total_contribution
            u_seg_new.append(u_n_seg_current + dt * total_contribution)

        return u_seg_new

    def _solve_imex_stage_u(
        self, u_n: np.ndarray, u_prev: np.ndarray, dt: float, stage: int
    ) -> np.ndarray:
        """求解IMEX-RK阶段 - 基于解值操作

        根据Doc/Time_Scheme.md 3.3节:
        阶段1: [V - γΔt(L1 + L2⊙F(u^n))] β^(1) = u^n + γΔt N(u^n)
        阶段2: [V - γΔt(L1 + L2⊙F(u^(1)))] β^(2) = u^n + 显式项
        """

        # 准备阶段数据 - 将全局u值转换为段值列表
        u_n_segments = []
        u_prev_segments = []

        start_idx = 0
        for seg_idx in range(self.ns):
            n_seg_points = len(self.data["x_segments_norm"][seg_idx])
            end_idx = start_idx + n_seg_points

            u_n_segments.append(u_n[start_idx:end_idx] if u_n is not None else None)
            u_prev_segments.append(
                u_prev[start_idx:end_idx] if u_prev is not None else None
            )

            start_idx = end_idx

        stage_data = {
            "u_n_seg": u_n_segments,
            "u_prev_seg": u_prev_segments,
            "dt": dt,
            "stage": stage,
            "gamma": self.gamma,
        }

        # 使用base_fitter的fit方法求解阶段系数
        coeffs_stage = self.fit(**stage_data)

        # 将系数转换为解值
        u_stage, u_stage_segments = self.construct(
            self.data, self._current_model, coeffs_stage
        )

        return u_stage,u_stage_segments

    def _imex_final_update_u(
        self, u_n: np.ndarray, U_stage1: np.ndarray, U_stage2: np.ndarray, dt: float
    ) -> np.ndarray:
        """IMEX-RK最终更新步骤 - 直接计算u^{n+1}

        根据Doc/Time_Scheme.md 3.4节公式:
        u^{n+1} = u^n + Δt/2 * [L1(U^(1)) + L1(U^(2)) + N(U^(1)) + N(U^(2)) + L2⊙F(U^(1)) + L2⊙F(U^(2))]
        """

        # 计算各阶段的算子贡献
        total_contribution = np.zeros_like(u_n)

        for stage_idx, (U_stage, weight) in enumerate(
            [(U_stage1, self.b[0]), (U_stage2, self.b[1])]
        ):
            stage_contribution = np.zeros_like(u_n)

            # 按段计算贡献
            start_idx = 0
            for segment_idx in range(self.ns):
                n_points = len(self.data["x_segments_norm"][segment_idx])
                end_idx = start_idx + n_points

                u_seg = U_stage[start_idx:end_idx]

                # L1项
                if self.has_operator("L1"):
                    L1_seg = self._linear_operators[segment_idx].get("L1", None)
                    if L1_seg is not None:
                        stage_contribution[start_idx:end_idx] += L1_seg @ u_seg

                # N项 (非线性，直接以u值作为输入)
                if self.has_operator("N"):
                    N_vals = self.N_func(self._features[segment_idx], u_seg)
                    if hasattr(N_vals, "__len__") and len(N_vals) == n_points:
                        stage_contribution[start_idx:end_idx] += N_vals
                    else:
                        stage_contribution[start_idx:end_idx] += N_vals

                # L2⊙F项
                if self.has_operator("L2") and self.has_operator("F"):
                    L2_seg = self._linear_operators[segment_idx].get("L2", None)
                    if L2_seg is not None:
                        F_vals = self.F_func(self._features[segment_idx], u_seg)
                        if hasattr(F_vals, "__len__") and len(F_vals) == n_points:
                            stage_contribution[start_idx:end_idx] += L2_seg @ (
                                F_vals * u_seg
                            )
                        else:
                            stage_contribution[start_idx:end_idx] += F_vals * (
                                L2_seg @ u_seg
                            )

                start_idx = end_idx

            # 加权累加
            total_contribution += weight * stage_contribution

        # 最终更新: u^{n+1} = u^n + Δt * total_contribution
        u_new = u_n + dt * total_contribution

        return u_new

    def _build_segment_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建段的IMEX-RK雅可比矩阵

        参考linear_pde_solver的设计模式，为每个段构造相应的线性系统
        """

        n_points = len(self.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.dgN

        L, r = self._build_stage_jacobian(segment_idx, **kwargs)

        return L, r

    def _build_stage_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建IMEX-RK阶段的段雅可比矩阵

        根据Doc/Time_Scheme.md 3.3节实现，完全参考linear_pde_solver的设计模式
        """

        n_points = len(self.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.dgN

        stage = kwargs.get("stage", 1)
        dt = kwargs.get("dt", 0.01)
        gamma = kwargs.get("gamma", self.gamma)

        # 获取预编译的线性算子 - 与linear_pde_solver相同的结构
        L1_operators = self._linear_operators[segment_idx].get("L1", None)
        L2_operators = self._linear_operators[segment_idx].get("L2", None)

        # 初始化L和b，完全采用linear_pde_solver的模式
        L = L1_operators  # 直接使用预编译的算子列表
        b = np.zeros((ne, n_points))

        # 修改L以包含IMEX-RK的隐式项
        if L is not None:
            # 创建修改后的L列表
            L_modified = []
            
            for eq_idx in range(ne):
                # 基础算子矩阵
                L_base = L[eq_idx].copy()

                # 添加隐式项: -γΔt * L1
                L_base = L_base - gamma * dt * L[eq_idx]

                # 添加L2*F项: -γΔt * diag(F) @ L2
                if L2_operators is not None and self.has_operator("F"):
                    # 获取当前段的u值
                    u_n_seg_list = kwargs.get("u_n_seg", [])
                    u_prev_seg_list = kwargs.get("u_prev_seg", [])

                    if (segment_idx < len(u_n_seg_list) and 
                        u_n_seg_list[segment_idx] is not None):
                        
                        u_n_seg = u_n_seg_list[segment_idx]
                        u_prev_seg = (
                            u_prev_seg_list[segment_idx]
                            if (segment_idx < len(u_prev_seg_list) and 
                                u_prev_seg_list[segment_idx] is not None)
                            else u_n_seg
                        )

                        # 选择u值 (阶段1用u_n，阶段2用u_prev)
                        u_seg_for_F = u_n_seg if stage == 1 else u_prev_seg

                        # 计算F值
                        F_vals = self.F_func(self._features[segment_idx], u_seg_for_F)
                        F_vals_eq = F_vals[:, eq_idx]

                        # 应用: L_base -= γΔt * diag(F) @ L2
                        L_base -= gamma * dt * np.diag(F_vals_eq) @ L2_operators[eq_idx]

                L_modified.append(L_base)
            
            L = L_modified

        # 构造右端向量 - 与linear_pde_solver相同的方式处理source
        r_seg = self._build_stage_rhs(segment_idx, stage, **kwargs)
        for i in range(ne):
            if r_seg.ndim > 1:
                b[i, :] = r_seg[:, i]
            else:
                # 如果只有一个方程，r_seg是1D向量
                b[i, :] = r_seg

        # 重新组织矩阵和向量，与linear_pde_solver完全一致
        L_final = np.vstack([L[i] for i in range(ne)])
        b_final = np.vstack([b[i].reshape(-1, 1) for i in range(ne)])

        return L_final, b_final

    def _build_stage_rhs(self, segment_idx: int, stage: int, **kwargs) -> np.ndarray:
        """构建IMEX-RK阶段的右端向量

        根据Doc/Time_Scheme.md 3.3节构建不同阶段的右端项
        """

        n_points = len(self.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs

        dt = kwargs.get("dt", 0.01)
        coeffs_n = kwargs.get("coeffs_n", np.zeros(self.dgN))

        # 当前段在全局系数中的索引
        if hasattr(coeffs_n, "shape") and len(coeffs_n.shape) > 2:
            coeffs_seg_n = coeffs_n[segment_idx, :, :].flatten()
        else:
            coeffs_seg_n = coeffs_n.flatten()[: self.dgN]

        # 基础特征矩阵
        features = self._features[segment_idx][0]

        # 初始化右端向量
        rhs = np.zeros((n_points, ne))

        if stage == 1:
            # 阶段1: RHS = u^n + γΔt N(u^n)
            gamma = kwargs.get("gamma", self.gamma)

            for eq_idx in range(ne):
                start_idx = eq_idx * self.dgN
                end_idx = (eq_idx + 1) * self.dgN
                coeffs_eq = (
                    coeffs_seg_n[start_idx:end_idx]
                    if len(coeffs_seg_n) > end_idx
                    else coeffs_seg_n[: self.dgN]
                )

                # 基础项: u^n
                rhs[:, eq_idx] = features @ coeffs_eq

                # 添加 γΔt N(u^n) 项
                if self.has_operator("N"):
                    u_n_seg = features @ coeffs_eq  # 当前段的u^n值
                    N_vals = self.N_func(self._features[segment_idx], u_n_seg)
                    if hasattr(N_vals, "__len__") and len(N_vals) == n_points:
                        rhs[:, eq_idx] += gamma * dt * N_vals
                    else:
                        rhs[:, eq_idx] += gamma * dt * N_vals

        elif stage == 2:
            # 阶段2: RHS = u^n + 显式项
            coeffs_prev = kwargs.get("coeffs_prev", coeffs_seg_n)
            if hasattr(coeffs_prev, "shape") and len(coeffs_prev.shape) > 2:
                coeffs_seg_prev = coeffs_prev[segment_idx, :, :].flatten()
            else:
                coeffs_seg_prev = coeffs_prev.flatten()[: self.dgN]

            gamma = kwargs.get("gamma", self.gamma)

            for eq_idx in range(ne):
                start_idx = eq_idx * self.dgN
                end_idx = (eq_idx + 1) * self.dgN

                # 基础项: u^n
                coeffs_eq_n = (
                    coeffs_seg_n[start_idx:end_idx]
                    if len(coeffs_seg_n) > end_idx
                    else coeffs_seg_n[: self.dgN]
                )
                rhs[:, eq_idx] = features @ coeffs_eq_n

                # 显式项: Δt(1-2γ)[L1(U^(1)) + L2(U^(1))F(U^(1))] + Δt(1-γ)N(U^(1))
                coeffs_eq_prev = (
                    coeffs_seg_prev[start_idx:end_idx]
                    if len(coeffs_seg_prev) > end_idx
                    else coeffs_seg_prev[: self.dgN]
                )

                # L1项
                if self.has_operator("L1"):
                    L1_seg = self._linear_operators[segment_idx].get("L1", None)
                    if L1_seg is not None:
                        u_prev = features @ coeffs_eq_prev
                        rhs[:, eq_idx] += dt * (1 - 2 * gamma) * (L1_seg @ u_prev)

                # N项
                if self.has_operator("N"):
                    u_prev = features @ coeffs_eq_prev  # 转换为解值
                    N_vals = self.N_func(self._features[segment_idx], u_prev)
                    if hasattr(N_vals, "__len__") and len(N_vals) == n_points:
                        rhs[:, eq_idx] += dt * (1 - gamma) * N_vals
                    else:
                        rhs[:, eq_idx] += dt * (1 - gamma) * N_vals

        return rhs


    def _build_update_rhs(self, segment_idx: int, eq_idx: int, **kwargs) -> np.ndarray:
        """构建最终更新步骤的右端向量

        根据Doc/Time_Scheme.md 3.4节: u^{n+1} = u^n + Δt/2 * [所有项的和]
        """

        n_points = len(self.data["x_segments_norm"][segment_idx])

        dt = kwargs.get("dt", 0.01)
        coeffs_n = kwargs.get("coeffs_n", np.zeros(self.dgN))
        coeffs_stage1 = kwargs.get("coeffs_stage1", coeffs_n)
        coeffs_stage2 = kwargs.get("coeffs_stage2", coeffs_n)
        b_weights = kwargs.get("b_weights", self.b)

        # 特征矩阵
        features = self._features[segment_idx][0]

        # 初始值: u^n
        if hasattr(coeffs_n, "shape") and len(coeffs_n.shape) > 2:
            coeffs_seg_n = coeffs_n[segment_idx, eq_idx, :]
        else:
            start_idx = eq_idx * self.dgN
            end_idx = (eq_idx + 1) * self.dgN
            coeffs_seg_n = (
                coeffs_n.flatten()[start_idx:end_idx]
                if len(coeffs_n.flatten()) > end_idx
                else coeffs_n.flatten()[: self.dgN]
            )

        u_n_seg = features @ coeffs_seg_n

        # 计算所有阶段的贡献
        total_contribution = np.zeros(n_points)

        for stage_idx, (coeffs_stage, weight) in enumerate(
            [(coeffs_stage1, b_weights[0]), (coeffs_stage2, b_weights[1])]
        ):
            if coeffs_stage is not None:
                # 获取阶段系数
                if hasattr(coeffs_stage, "shape") and len(coeffs_stage.shape) > 2:
                    coeffs_seg_stage = coeffs_stage[segment_idx, eq_idx, :]
                else:
                    start_idx = eq_idx * self.dgN
                    end_idx = (eq_idx + 1) * self.dgN
                    coeffs_seg_stage = (
                        coeffs_stage.flatten()[start_idx:end_idx]
                        if len(coeffs_stage.flatten()) > end_idx
                        else coeffs_stage.flatten()[: self.dgN]
                    )

                # L1项
                if self.has_operator("L1"):
                    L1_seg = self._linear_operators[segment_idx].get("L1", None)
                    if L1_seg is not None:
                        u_stage = features @ coeffs_seg_stage
                        total_contribution += weight * (L1_seg @ u_stage)

                # N项 (非线性)
                if self.has_operator("N"):
                    u_stage = features @ coeffs_seg_stage
                    N_vals = self.N_func(self._features[segment_idx], u_stage)
                    if hasattr(N_vals, "__len__") and len(N_vals) == n_points:
                        total_contribution += weight * N_vals
                    else:
                        total_contribution += weight * N_vals

                # L2*F项
                if self.has_operator("L2") and self.has_operator("F"):
                    L2_seg = self._linear_operators[segment_idx].get("L2", None)
                    if L2_seg is not None:
                        u_stage = features @ coeffs_seg_stage
                        F_vals = self.F_func(self._features[segment_idx], u_stage)
                        if hasattr(F_vals, "__len__") and len(F_vals) == n_points:
                            total_contribution += weight * (L2_seg @ (F_vals * u_stage))
                        else:
                            total_contribution += weight * F_vals * (L2_seg @ u_stage)

        # 最终结果: u^n + Δt * total_contribution
        rhs = u_n_seg + dt * total_contribution

        return rhs

    def solve_time_step(self, u_n: np.ndarray, dt: float, **kwargs) -> np.ndarray:
        """求解一个时间步 - 使用IMEX-RK(2,2,2)方法"""

        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        return self.imex_rk_time_step(u_n, dt)

    def get_butcher_tableau(self) -> Dict[str, np.ndarray]:
        """获取IMEX-RK(2,2,2) Butcher表"""
        return {
            "A_implicit": self.A_imp,
            "A_explicit": self.A_exp,
            "b_weights": self.b,
            "gamma": self.gamma,
        }

    def _build_jacobian_nonlinear(
        self, data: Dict, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建雅可比矩阵和残差向量，支持IMEX-RK(2,2,2)和BDF格式"""
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
                **kwargs,  # 传递额外参数如stage等
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

    def validate_operators(self) -> Dict[str, bool]:
        """验证算子配置是否正确"""

        validation = {
            "L1_exists": self.has_operator("L1"),
            "L2_exists": self.has_operator("L2"),
            "N_exists": self.has_operator("N"),
            "F_exists": self.has_operator("F"),
            "precompiled": self._precompiled,
        }

        # 检查IMEX-RK的最小要求
        validation["imex_ready"] = validation["precompiled"] and (
            validation["L1_exists"] or validation["L2_exists"]
        )

        return validation

    def get_time_step_info(self) -> Dict[str, Any]:
        """获取时间步进信息"""

        info = {
            "method": "IMEX-RK(2,2,2)",
            "stages": 2,
            "order": 2,
            "stability": "A-stable for stiff terms, explicit for non-stiff terms",
            "gamma": self.gamma,
            "operators": {
                "L1": "Implicit linear operator (e.g., diffusion)",
                "L2": "Implicit linear operator with nonlinear coefficient",
                "N": "Explicit nonlinear operator (e.g., reaction)",
                "F": "Nonlinear coefficient function",
            },
            "equation_form": "du/dt = L1(u) + N(u) + L2(u)*F(u)",
        }

        return info

    def create_time_integrator(
        self, time_scheme: str = "imex_rk_222"
    ) -> Dict[str, Any]:
        """创建时间积分器配置"""

        if time_scheme == "imex_rk_222":
            return {
                "scheme": "IMEX-RK(2,2,2)",
                "stages": 2,
                "order": 2,
                "A_implicit": self.A_imp,
                "A_explicit": self.A_exp,
                "b_weights": self.b,
                "gamma": self.gamma,
                "step_function": self.imex_rk_time_step,
            }
        else:
            raise ValueError(f"Unsupported time scheme: {time_scheme}")

    def print_time_scheme_summary(self):
        """打印时间格式摘要"""

        validation = self.validate_operators()
        info = self.get_time_step_info()

        print("=== IMEX-RK(2,2,2) Time Integration Summary ===")
        print(f"Method: {info['method']}")
        print(f"Stages: {info['stages']}, Order: {info['order']}")
        print(f"Gamma parameter: {self.gamma:.6f}")
        print(f"Equation form: {info['equation_form']}")
        print("\nOperator Status:")
        print(f"  L1 (implicit linear): {'✓' if validation['L1_exists'] else '✗'}")
        print(f"  L2 (implicit with F): {'✓' if validation['L2_exists'] else '✗'}")
        print(f"  N  (explicit nonlinear): {'✓' if validation['N_exists'] else '✗'}")
        print(f"  F  (nonlinear coefficient): {'✓' if validation['F_exists'] else '✗'}")
        print(f"\nPrecompilation status: {'✓' if validation['precompiled'] else '✗'}")
        print(f"Ready for time integration: {'✓' if validation['imex_ready'] else '✗'}")
        print("=" * 50)

    def estimate_stable_dt(
        self, u_current: np.ndarray, safety_factor: float = 0.8
    ) -> float:
        """估算稳定的时间步长

        根据Doc/Time_Scheme.md中的稳定性约束:
        Δt ≤ C_stability / max(ρ(L1), ρ(L2)*max|F'(u)|, max|N'(u)|)
        """

        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        max_eigenvalue = 0.0

        # 估算L1的谱半径
        if self.has_operator("L1"):
            for segment_idx in range(self.ns):
                L1_op = self._linear_operators[segment_idx].get("L1", None)
                if L1_op is not None:
                    try:
                        eigenvals = np.linalg.eigvals(L1_op)
                        max_eigenvalue = max(max_eigenvalue, np.max(np.real(eigenvals)))
                    except:
                        # 如果特征值计算失败，使用保守估计
                        max_eigenvalue = max(max_eigenvalue, np.max(np.abs(L1_op)))

        # 估算L2*F'项（简化处理）
        if self.has_operator("L2") and self.has_operator("F"):
            max_eigenvalue = max(max_eigenvalue, 1.0)  # 保守估计

        # 估算N'项（简化处理）
        if self.has_operator("N"):
            max_eigenvalue = max(max_eigenvalue, 1.0)  # 保守估计

        if max_eigenvalue <= 0:
            return 0.1  # 默认值

        # IMEX-RK(2,2,2)的稳定性常数约为1.5
        C_stability = 1.5
        dt_stable = safety_factor * C_stability / max_eigenvalue

        return dt_stable
