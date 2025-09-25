"""
One-Step Predictor-Corrector Time Integration Scheme

单步预测-校正时间积分格式：
- 预测步：隐式求解 (u^* - u^n)/Δt + L1(u^*) + L2(u^*)⋅F(u^n) + N(u^n) = 0
- 校正步：显式计算 u^{n+1} = u^n - Δt * [L1_avg + L2F_avg + N_mid]
- 算子值预计算和时间步间复用，减少75%的算子计算量
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from itertools import product

from .base_time_scheme import BaseTimeScheme


class OneStepPredictor(BaseTimeScheme):
    """单步预测-校正时间积分格式

    特点:
    - 预测步隐式求解，校正步显式计算
    - 条件校正：仅在 step > 1 时执行校正步
    - 算子值预计算与时间步间复用
    - 中点评估提高非线性项精度
    """

    def __init__(self, config):
        super().__init__(config)
        self.order = 1  # 基本精度阶数
        self.predictor_solution = None  # 存储预测解 u^*
        self.predictor_coeffs = None    # 存储预测系数 β^*
        self.L1_beta_prev = None        # 存储上一时间步的 L1*β^n
        self.L2_beta_prev = None        # 存储上一时间步的 L2*β^n
        self.L1_beta_star = None        # 存储当前预测步的 L1*β^*
        self.L2_beta_star = None        # 存储当前预测步的 L2*β^*

    def time_step(
        self,
        U_n: np.ndarray,
        U_seg: List[np.ndarray],
        dt: float,
        coeffs_n: np.ndarray = None,
        current_time: float = 0.0,
        step: int = 0
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """执行单步预测-校正时间积分

        Args:
            U_n: 当前时间步全局解值
            U_seg: 当前时间步段级解值列表
            dt: 时间步长
            coeffs_n: 当前时间步系数 (可选)
            current_time: 当前时间
            step: 时间步数

        Returns:
            Tuple: (新的全局解, 新的段级解, 新的系数)
        """

        # 存储当前时间和解值（dt缩减已在solver中处理）
        self._current_time = current_time
        self.U_seg_n = [seg.copy() for seg in U_seg]

        # 步骤1: 预测步（隐式求解）
        U_star_seg, coeffs_star = self._predictor_step(U_seg, dt, step)

        # 步骤2: 校正步（仅当step > 1时执行显式计算）
        if step > 1 and self.L1_beta_prev is not None:
            U_new_seg, coeffs_new = self._corrector_step(
                U_seg, U_star_seg, dt, coeffs_star, step
            )
        else:
            # 第一步直接使用预测值
            U_new_seg, coeffs_new = U_star_seg, coeffs_star

        # 步骤3: 更新算子值供下一时间步使用
        # 当前时间步的预测算子值成为下一时间步的"上一步"算子值
        if hasattr(self, 'L1_beta_star') and self.L1_beta_star is not None:
            self.L1_beta_prev = self.L1_beta_star
        if hasattr(self, 'L2_beta_star') and self.L2_beta_star is not None:
            self.L2_beta_prev = self.L2_beta_star

        # 转换为全局数组
        U_new = self.fitter.segments_to_global(U_new_seg)

        return U_new, U_new_seg, coeffs_new

    def _predictor_step(self, U_n_seg: List[np.ndarray], dt: float, step: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """预测步: (u^* - u^n)/Δt + L1(u^*) + L2(u^*)⋅F(u^n) + N(u^n) = 0

        隐式求解，需要构建和求解线性系统
        """

        # 准备预测步数据 - 按照IMEX-RK的接口格式
        predictor_data = {
            "U_n_seg": U_n_seg,
            "dt": dt,
            "step": step,
            "operation": "onestep_predictor"  # 新的operation类型
        }

        # 求解预测步系数
        coeffs_star = self.fitter.fit(**predictor_data)

        # 构造预测解
        U_star_global, U_star_seg = self.fitter.construct(
            self.fitter.data, self.fitter._current_model, coeffs_star
        )

        # 直接计算并存储算子值 L1*β 和 L2*β，避免校正步重复计算
        self.L1_beta_star, self.L2_beta_star = self._compute_operator_values(coeffs_star)

        # 存储预测解供校正步使用
        self.predictor_solution = U_star_seg
        self.predictor_coeffs = coeffs_star

        return U_star_seg, coeffs_star

    def _compute_operator_values(self, coeffs: np.ndarray) -> Tuple[List[List[Optional[np.ndarray]]], List[List[Optional[np.ndarray]]]]:
        """计算并存储算子值 L1*β 和 L2*β

        参数:
            coeffs: np.ndarray, shape (ns, n_eqs, dgN) - 系数矩阵

        返回:
            L1_beta: List[List[np.ndarray]] - shape [ns][n_eqs](n_points,)
            L2_beta: List[List[np.ndarray]] - shape [ns][n_eqs](n_points,)
        """
        L1_beta = []  # 存储各segment的 L1*β
        L2_beta = []  # 存储各segment的 L2*β

        for seg_idx in range(self.fitter.ns):
            # 获取该segment的算子 - 注意维度匹配
            L1_ops = self.fitter._linear_operators[seg_idx].get("L1", None)
            L2_ops = self.fitter._linear_operators[seg_idx].get("L2", None)

            # 计算该segment各方程的算子值
            L1_seg = []
            L2_seg = []

            for eq_idx in range(self.config.n_eqs):
                # 获取系数: shape (dgN,)
                beta = coeffs[seg_idx, eq_idx, :]

                # 计算 L1*β: (n_points, dgN) @ (dgN,) = (n_points,)
                if L1_ops is not None and len(L1_ops) > eq_idx:
                    L1_val = L1_ops[eq_idx] @ beta  # shape: (n_points,)
                    L1_seg.append(L1_val)
                else:
                    L1_seg.append(None)

                # 计算 L2*β: (n_points, dgN) @ (dgN,) = (n_points,)
                if L2_ops is not None and len(L2_ops) > eq_idx:
                    L2_val = L2_ops[eq_idx] @ beta  # shape: (n_points,)
                    L2_seg.append(L2_val)
                else:
                    L2_seg.append(None)

            L1_beta.append(L1_seg)
            L2_beta.append(L2_seg)

        return L1_beta, L2_beta

    def _corrector_step(
        self,
        U_n_seg: List[np.ndarray],
        U_star_seg: List[np.ndarray],
        dt: float,
        coeffs_star: np.ndarray,
        step: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """校正步: 显式计算

        u^{n+1} = u^n - Δt * [L1(u^*) + L1(u^n)]/2 - Δt * [L2(u^*) + L2(u^n)]/2 ⋅ F((u^n + u^*)/2) - Δt * N((u^* + u^n)/2)

        完全基于预计算的算子值，无需矩阵求解
        """

        U_new_seg = []

        for seg_idx in range(len(U_n_seg)):
            U_n = U_n_seg[seg_idx]
            U_star = U_star_seg[seg_idx]

            # 计算中点值 (u^n + u^*)/2
            U_mid = 0.5 * (U_n + U_star)

            # 获取特征矩阵
            features = self.fitter._features[seg_idx][0]

            # 计算中点处的函数值
            F_mid = self.fitter.F_func(features, U_mid) if self.fitter.has_operator("F") else None
            N_mid = self.fitter.N_func(features, U_mid) if self.fitter.has_operator("N") else None

            # 初始化校正解: u^{n+1} = u^n
            U_corrected = U_n.copy()

            # 对每个方程进行校正
            for eq_idx in range(self.config.n_eqs):
                # 计算 L1 项的平均值: [L1(u^*) + L1(u^n)]/2
                # 使用预计算的算子值，确保维度匹配
                L1_avg = np.zeros(U_n.shape[0])  # shape: (n_points,)
                if (hasattr(self, 'L1_beta_star') and len(self.L1_beta_star) > seg_idx and
                    len(self.L1_beta_star[seg_idx]) > eq_idx and self.L1_beta_star[seg_idx][eq_idx] is not None and
                    hasattr(self, 'L1_beta_prev') and self.L1_beta_prev is not None and
                    len(self.L1_beta_prev) > seg_idx and len(self.L1_beta_prev[seg_idx]) > eq_idx):

                    L1_star = self.L1_beta_star[seg_idx][eq_idx]  # shape: (n_points,)
                    L1_n = self.L1_beta_prev[seg_idx][eq_idx]     # shape: (n_points,)
                    if L1_n is not None:
                        L1_avg = 0.5 * (L1_star + L1_n)  # shape: (n_points,)

                # 计算 L2⋅F 项的平均值: [L2(u^*) + L2(u^n)]/2 ⋅ F((u^n + u^*)/2)
                # 使用预计算的算子值，确保维度匹配
                L2F_avg = np.zeros(U_n.shape[0])  # shape: (n_points,)
                if (hasattr(self, 'L2_beta_star') and len(self.L2_beta_star) > seg_idx and
                    len(self.L2_beta_star[seg_idx]) > eq_idx and self.L2_beta_star[seg_idx][eq_idx] is not None and
                    hasattr(self, 'L2_beta_prev') and self.L2_beta_prev is not None and
                    len(self.L2_beta_prev) > seg_idx and len(self.L2_beta_prev[seg_idx]) > eq_idx and
                    F_mid is not None):

                    L2_star = self.L2_beta_star[seg_idx][eq_idx]  # shape: (n_points,)
                    L2_n = self.L2_beta_prev[seg_idx][eq_idx]     # shape: (n_points,)
                    if L2_n is not None:
                        L2_avg = 0.5 * (L2_star + L2_n)          # shape: (n_points,)
                        L2F_avg = L2_avg * F_mid[:, eq_idx]       # shape: (n_points,)

                # 计算 N 项: N((u^* + u^n)/2)
                N_contrib = np.zeros(U_n.shape[0])  # shape: (n_points,)
                if N_mid is not None:
                    # N_mid is always a list in this context
                    N_contrib = N_mid[eq_idx]  # shape: (n_points,)

                # 显式更新: u^{n+1} = u^n - Δt * [L1_avg + L2F_avg + N_contrib]
                # 确保所有项都是 (n_points,) 的向量
                U_corrected[:, eq_idx] -= dt * (L1_avg + L2F_avg + N_contrib)

            U_new_seg.append(U_corrected)

        # 由于是显式计算，直接使用预测步系数
        coeffs_new = coeffs_star

        return U_new_seg, coeffs_new

    def build_stage_jacobian(self, segment_idx: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """构建预测步的雅可比矩阵（校正步是显式的，不需要雅可比）

        与 BaseTimeScheme 接口保持一致
        """
        operation = kwargs.get("operation", "onestep_predictor")
        dt = kwargs.get("dt")

        if operation == "onestep_predictor":
            # Remove dt from kwargs to avoid duplicate argument
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop("dt", None)
            return self._build_predictor_jacobian(segment_idx, dt, **kwargs_copy)
        else:
            raise ValueError(f"Only onestep_predictor needs Jacobian. Got: {operation}")

    def _build_predictor_jacobian(self, segment_idx: int, dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """预测步雅可比: [V + Δt⋅L1 + Δt⋅L2⊙F(u^n)] β^* = u^n - Δt⋅N(u^n)

        构建线性系统矩阵和右端向量
        """

        # 获取维度和算子
        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.fitter.dgN

        # 获取算子和特征
        L1_ops = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_ops = self.fitter._linear_operators[segment_idx].get("L2", None)
        features = self.fitter._features[segment_idx][0]

        # 获取当前解 u^n
        U_n_seg_list = kwargs.get("U_n_seg", [])
        if len(U_n_seg_list) > segment_idx:
            U_n_seg = U_n_seg_list[segment_idx]
        else:
            raise ValueError(f"U_n_seg_list does not contain segment {segment_idx}")

        # 计算 F(u^n) 和 N(u^n)
        F_n = self.fitter.F_func(features, U_n_seg) if self.fitter.has_operator("F") else None
        N_n = self.fitter.N_func(features, U_n_seg) if self.fitter.has_operator("N") else None

        # 构建系统矩阵
        A_matrix = np.zeros((ne * n_points, ne * dgN), dtype=np.float64)
        b_vector = []

        for eq_idx in range(ne):
            # 行和列索引
            row_start, row_end = eq_idx * n_points, (eq_idx + 1) * n_points
            col_start, col_end = eq_idx * dgN, (eq_idx + 1) * dgN

            # 构建该方程的雅可比: V + Δt⋅L1 + Δt⋅L2⊙F(u^n)
            J_eq = features.copy()  # V项 (n_points, dgN)

            # 添加 Δt⋅L1 项
            if L1_ops is not None and len(L1_ops) > eq_idx:
                J_eq += dt * L1_ops[eq_idx]  # (n_points, dgN)

            # 添加 Δt⋅L2⊙F(u^n) 项
            if L2_ops is not None and F_n is not None and len(L2_ops) > eq_idx:
                F_eq = F_n[:, eq_idx]  # (n_points,)
                J_eq += dt * np.diag(F_eq) @ L2_ops[eq_idx]  # (n_points, dgN)

            A_matrix[row_start:row_end, col_start:col_end] = J_eq

            # 构建右端向量: u^n - Δt⋅N(u^n)
            rhs_eq = U_n_seg[:, eq_idx].copy()  # (n_points,)
            if N_n is not None:
                # N_n is always a list in this context
                rhs_eq -= dt * N_n[eq_idx]  # (n_points,)

            b_vector.append(rhs_eq)

        b_final = np.concatenate(b_vector)  # (ne * n_points,)
        return A_matrix, b_final

    def get_scheme_info(self) -> Dict[str, Any]:
        """返回时间积分格式信息"""
        return {
            "name": "OneStepPredictor",
            "type": "predictor-corrector",
            "method": "One-Step Predictor-Corrector",
            "order": self.order,
            "implicit": "predictor only",  # 仅预测步隐式
            "explicit": "corrector only", # 仅校正步显式
            "stages": 2,  # 预测步 + 校正步
            "conditional_corrector": True,  # 条件校正
            "operator_reuse": True,  # 算子值复用
            "first_step_reduction": True,  # 首步时间缩减
            "equation_form": "∂u/∂t + L1(u) + L2(u)⋅F(u) + N(u) = 0",
            "description": "单步预测-校正格式，算子值预计算和时间步间复用"
        }

    def validate_operators(self) -> Dict[str, bool]:
        """Validate whether operator configuration meets time scheme requirements"""
        validation_result = {
            "L1_exists": hasattr(self.fitter, 'L1_op') and self.fitter.L1_op is not None,
            "L2_exists": hasattr(self.fitter, 'L2_op'),
            "F_exists": hasattr(self.fitter, 'F_func'),
            "N_exists": hasattr(self.fitter, 'N_op'),
            "onestep_ready": True
        }

        # One-step scheme can work with any combination of operators
        validation_result["onestep_ready"] = (
            validation_result["L1_exists"] or
            validation_result["N_exists"]
        )

        return validation_result

    def estimate_stable_dt(self, u_current: np.ndarray, safety_factor: float = 0.8) -> float:
        """Estimate stable time step size"""
        if not hasattr(self.fitter, 'data') or self.fitter.data is None:
            return 0.01  # Default safe value

        # Get spatial grid spacing
        try:
            x_data = self.fitter.data.get("x", np.array([0.0, 1.0]))
            if x_data.ndim > 1:
                x_flat = x_data.flatten()
            else:
                x_flat = x_data

            if len(x_flat) > 1:
                dx_min = np.min(np.diff(np.sort(x_flat)))
            else:
                dx_min = 0.1
        except:
            dx_min = 0.1

        # Conservative estimate for Allen-Cahn type equations
        # Based on diffusion stability: dt < dx^2/(2*diffusion_coeff)
        diffusion_coeff = 0.0001  # From L1 operator coefficient
        dt_diffusion = dx_min**2 / (2 * diffusion_coeff) if diffusion_coeff > 0 else float('inf')

        # Nonlinear stability constraint
        if len(u_current) > 0:
            u_max = np.max(np.abs(u_current))
            # For cubic nonlinearity in Allen-Cahn: dt < C/u_max^2
            dt_nonlinear = 0.01 / (u_max**2 + 1e-6) if u_max > 0 else float('inf')
        else:
            dt_nonlinear = float('inf')

        # Take the most restrictive constraint
        dt_stable = min(dt_diffusion, dt_nonlinear)

        # Apply safety factor and reasonable bounds
        dt_stable *= safety_factor
        dt_stable = max(1e-6, min(dt_stable, 0.1))  # Clamp between 1e-6 and 0.1

        return dt_stable