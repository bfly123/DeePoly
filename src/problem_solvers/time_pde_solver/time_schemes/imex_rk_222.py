"""IMEX-RK(2,2,2) time integration scheme"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .base_time_scheme import BaseTimeScheme


class ImexRK222(BaseTimeScheme):
    """IMEX-RK(2,2,2) time integration scheme implementation"""

    def __init__(self, config):
        super().__init__(config)

        # IMEX-RK(2,2,2) parameters (according to formula 3.3)
        self.gamma = (2 - np.sqrt(2)) / 2  # ≈ 0.2928932

        # Butcher tableau coefficients
        self.A_imp = np.array([[self.gamma, 0], [1 - 2 * self.gamma, self.gamma]])
        self.A_exp = np.array([[0, 0], [1 - self.gamma, 0]])
        self.b = np.array([0.5, 0.5])

        # Storage for stage solutions
        self.stage_solutions = {}

        # Maintain current segment-level solution values to avoid recomputation per segmentation
        self.u_seg_current = (
            None  # List[np.ndarray] - current timestep segment-level solution values
        )
        self.u_seg_prev = (
            None  # List[np.ndarray] - previous timestep segment-level solution values
        )

    def time_step(
        self, u_n: np.ndarray, u_seg: List[np.ndarray], dt: float, coeffs_n: np.ndarray = None
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute IMEX-RK(2,2,2) time step"""

        # Initialize or update current segment-level solution values
        self.u_seg_current = u_seg.copy()

        # Stage 1: Solve U^(1)
        u_seg_stage1, coeffs_stage1 = self._solve_imex_stage_u_seg(
            self.u_seg_current, None, dt, stage=1, coeffs_n=coeffs_n
        )

        # Stage 2: Solve U^(2)
        u_seg_stage2, coeffs_stage2 = self._solve_imex_stage_u_seg(
            self.u_seg_current, u_seg_stage1, dt, stage=2, coeffs_n=coeffs_n
        )

        # Final update: Calculate u^{n+1} (according to formula 3.4)
        u_seg_new = self._imex_final_update_u_seg(
            self.u_seg_current,
            u_seg_stage1,
            u_seg_stage2,
            dt,
            coeffs_stage1,
            coeffs_stage2,
        )

        # Update maintained segment-level solution values
        # self.u_seg_prev = self.u_seg_current
        # self.u_seg_current = u_seg_new

        # Convert back to global array
        u_new = self.fitter.segments_to_global(u_seg_new)

        return (
            u_new,
            u_seg_new,
            coeffs_stage2,
        )  # Return coefficients from the last stage

    def _solve_imex_stage_u_seg(
        self,
        u_n_seg: List[np.ndarray],
        u_prev_seg: List[np.ndarray],
        dt: float,
        stage: int,
        coeffs_n: np.ndarray = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """求解IMEX-RK阶段 - 直接使用段级解值操作"""

        # 准备阶段数据 - 直接传递段级数据
        stage_data = {
            "u_n_seg": u_n_seg,
            "u_prev_seg": u_prev_seg,
            "dt": dt,
            "stage": stage,
            "gamma": self.gamma,
            "operation": "imex_stage",
            "coeffs_n": coeffs_n,
        }

        # 使用base_fitter的fit方法求解阶段系数
        coeffs_stage = self.fitter.fit(**stage_data)

        # 将系数转换为解值
        u_stage_global, u_stage_segments = self.fitter.construct(
            self.fitter.data, self.fitter._current_model, coeffs_stage
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
        """IMEX-RK最终更新步骤 - 直接使用段级解值和系数"""

        u_seg_new = []

        for segment_idx in range(self.fitter.ns):
            n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
            u_n_seg_current = u_n_seg[segment_idx]

            # Calculate operator contributions from each stage
            total_contribution = np.zeros_like(u_n_seg_current)

            for stage_idx, (u_seg_stage, coeffs_stage, weight) in enumerate(
                [
                    (u_seg_stage1[segment_idx], coeffs_stage1, self.b[0]),
                    (u_seg_stage2[segment_idx], coeffs_stage2, self.b[1]),
                ]
            ):
                stage_contribution = np.zeros_like(u_n_seg_current)
                print(f"Debug stage_contribution.shape={stage_contribution.shape}")

                # Extract coefficients for current segment
                ne = self.config.n_eqs
                dgN = self.fitter.dgN

                for eq_idx in range(ne):
                    # Get coefficients for current segment and equation
                    print(f"Debug coeffs: stage_idx={stage_idx}, coeffs_stage.shape={coeffs_stage.shape}")
                    
                    # 处理不同的系数格式
                    if coeffs_stage.ndim == 3:  # (ns, ne, dgN)
                        beta_seg_stage = coeffs_stage[segment_idx, eq_idx, :]
                    elif coeffs_stage.ndim == 1:  # 展平的系数
                        start_idx = (segment_idx * ne + eq_idx) * dgN
                        end_idx = start_idx + dgN
                        beta_seg_stage = coeffs_stage[start_idx:end_idx]
                    else:
                        print(f"Unexpected coeffs_stage shape: {coeffs_stage.shape}")
                        beta_seg_stage = np.zeros(dgN)

                    # L1 term: L1 @ β^(i)
                    if self.fitter.has_operator("L1"):
                        L1_seg = self.fitter._linear_operators[segment_idx].get(
                            "L1", None
                        )
                        if L1_seg is not None:
                            print(f"Debug final: L1_seg.shape={L1_seg.shape}, beta_seg_stage.shape={beta_seg_stage.shape}")
                            # 如果L1_seg是3D，取第一个切片
                            if L1_seg.ndim == 3:
                                L1_seg_2d = L1_seg[0]  # 取第一个切片，变成2D
                            else:
                                L1_seg_2d = L1_seg
                            result = L1_seg_2d @ beta_seg_stage
                            # 确保结果维度与stage_contribution兼容
                            if result.ndim == 1 and stage_contribution.ndim == 2:
                                result = result.reshape(-1, 1)
                            stage_contribution += result

                # N term: N(u^(i)) - nonlinear, directly using u values as input
                if self.fitter.has_operator("N"):
                    N_vals = self.fitter.N_func(
                        self.fitter._features[segment_idx], u_seg_stage
                    )
                    if hasattr(N_vals, "__len__") and len(N_vals) == n_points:
                        stage_contribution += N_vals
                    else:
                        stage_contribution += N_vals

                # L2⊙F term: L2 @ (F(u^(i)) * β^(i))
                if self.fitter.has_operator("L2") and self.fitter.has_operator("F"):
                    L2_seg = self.fitter._linear_operators[segment_idx].get("L2", None)
                    if L2_seg is not None:
                        F_vals = self.fitter.F_func(
                            self.fitter._features[segment_idx], u_seg_stage
                        )

                        for eq_idx in range(ne):
                            beta_seg_stage = coeffs_stage[segment_idx, eq_idx, :]
                            if hasattr(F_vals, "__len__") and len(F_vals) == n_points:
                                stage_contribution += L2_seg @ (F_vals * beta_seg_stage)
                            else:
                                stage_contribution += F_vals * (L2_seg @ beta_seg_stage)

                # Weighted accumulation
                total_contribution += weight * stage_contribution

            # Final update: u^{n+1} = u^n + Δt * total_contribution
            u_seg_new.append(u_n_seg_current + dt * total_contribution)

        return u_seg_new

    def build_stage_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建IMEX-RK阶段的段雅可比矩阵"""

        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.fitter.dgN

        stage = kwargs.get("stage", 1)
        dt = kwargs.get("dt", 0.01)
        gamma = kwargs.get("gamma", self.gamma)

        # 获取预编译的线性算子 - 与linear_pde_solver相同的结构
        L1_operators = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)

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
                if L2_operators is not None and self.fitter.has_operator("F"):
                    # 获取当前段的u值
                    u_n_seg_list = kwargs.get("u_n_seg", [])
                    u_prev_seg_list = kwargs.get("u_prev_seg", [])

                    if (
                        segment_idx < len(u_n_seg_list)
                        and u_n_seg_list[segment_idx] is not None
                    ):

                        u_n_seg = u_n_seg_list[segment_idx]
                        u_prev_seg = (
                            u_prev_seg_list[segment_idx]
                            if (
                                segment_idx < len(u_prev_seg_list)
                                and u_prev_seg_list[segment_idx] is not None
                            )
                            else u_n_seg
                        )

                        # 选择u值 (阶段1用u_n，阶段2用u_prev)
                        u_seg_for_F = u_n_seg if stage == 1 else u_prev_seg

                        # 计算F值
                        F_vals = self.fitter.F_func(
                            self.fitter._features[segment_idx], u_seg_for_F
                        )
                        F_vals_eq = F_vals[:, eq_idx]

                        # 应用: L_base -= γΔt * diag(F) @ L2
                        L_base -= gamma * dt * np.diag(F_vals_eq) @ L2_operators[eq_idx]

                L_modified.append(L_base)

            L = L_modified

        # 构造右端向量 - 与linear_pde_solver相同的方式处理source
        # 从kwargs中移除stage以避免重复参数
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('stage', None)
        r_seg = self._build_stage_rhs(segment_idx, stage, **kwargs_copy)
        for i in range(ne):
            if r_seg.ndim > 1:
                b[i, :] = r_seg[:, i]
            else:
                # 如果只有一个方程，r_seg是1D向量
                b[i, :] = r_seg

        # 重新组织矩阵和向量，与linear_pde_solver完全一致
        L_final = np.vstack([L[i] for i in range(ne)])
        b_final = np.vstack([b[i].reshape(-1, 1) for i in range(ne)])

        return L_final, b_final.flatten()  # 展平为1D以匹配base_fitter期望

    def _build_stage_rhs(self, segment_idx: int, stage: int, **kwargs) -> np.ndarray:
        """构建IMEX-RK阶段的右端向量"""

        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs

        dt = kwargs.get("dt", 0.01)
        coeffs_n = kwargs.get("coeffs_n", np.zeros(self.fitter.dgN))

        # 当前段在全局系数中的索引 - 处理不同的系数数组结构
        if coeffs_n is not None:
            if coeffs_n.ndim == 3:  # (ns, ne, dgN)
                coeffs_seg_n = coeffs_n[segment_idx, :, :].flatten()
            elif coeffs_n.ndim == 1:  # 展平的系数数组
                # 计算当前段的起始索引
                start_idx = segment_idx * ne * self.fitter.dgN
                end_idx = start_idx + ne * self.fitter.dgN
                coeffs_seg_n = coeffs_n[start_idx:end_idx]
            else:
                coeffs_seg_n = np.zeros(ne * self.fitter.dgN)
        else:
            coeffs_seg_n = np.zeros(ne * self.fitter.dgN)

        # 基础特征矩阵
        features = self.fitter._features[segment_idx][0]

        # 初始化右端向量
        rhs = np.zeros((n_points, ne))

        if stage == 1:
            # 阶段1: RHS = u^n + γΔt N(u^n)
            gamma = kwargs.get("gamma", self.gamma)

            for eq_idx in range(ne):
                start_idx = eq_idx * self.fitter.dgN
                end_idx = (eq_idx + 1) * self.fitter.dgN
                coeffs_eq = coeffs_seg_n[start_idx:end_idx]

                # 基础项: u^n
                rhs[:, eq_idx] = features @ coeffs_eq

                # 添加 γΔt N(u^n) 项
                if self.fitter.has_operator("N"):
                    u_n_seg = features @ coeffs_eq  # 当前段的u^n值
                    N_vals = self.fitter.N_func(u_n_seg)
                    rhs[:, eq_idx] += gamma * dt * N_vals

        elif stage == 2:
            # 阶段2: RHS = u^n + 显式项
            coeffs_prev = kwargs.get("coeffs_prev", coeffs_seg_n)
            
            # 处理不同的系数数组结构
            if coeffs_prev is not None:
                if coeffs_prev.ndim == 3:  # (ns, ne, dgN)
                    coeffs_seg_prev = coeffs_prev[segment_idx, :, :].flatten()
                elif coeffs_prev.ndim == 1:  # 展平的系数数组
                    # 计算当前段的起始索引
                    start_idx_prev = segment_idx * ne * self.fitter.dgN
                    end_idx_prev = start_idx_prev + ne * self.fitter.dgN
                    coeffs_seg_prev = coeffs_prev[start_idx_prev:end_idx_prev]
                else:
                    coeffs_seg_prev = np.zeros(ne * self.fitter.dgN)
            else:
                coeffs_seg_prev = np.zeros(ne * self.fitter.dgN)

            gamma = kwargs.get("gamma", self.gamma)

            for eq_idx in range(ne):
                start_idx = eq_idx * self.fitter.dgN
                end_idx = (eq_idx + 1) * self.fitter.dgN

                # 基础项: u^n
                coeffs_eq_n = coeffs_seg_n[start_idx:end_idx]
                rhs[:, eq_idx] = features @ coeffs_eq_n

                # 显式项: Δt(1-2γ)[L1(U^(1)) + L2(U^(1))F(U^(1))] + Δt(1-γ)N(U^(1))
                coeffs_eq_prev = coeffs_seg_prev[start_idx:end_idx]

                # L1项
                if self.fitter.has_operator("L1"):
                    L1_seg = self.fitter._linear_operators[segment_idx].get("L1", None)
                    if L1_seg is not None:
                        print(f"Debug L1: L1_seg.shape={L1_seg.shape}, coeffs_eq_prev.shape={coeffs_eq_prev.shape}")
                        if coeffs_eq_prev.size > 0:
                            rhs[:, eq_idx] += dt * (1 - 2 * gamma) * (L1_seg @ coeffs_eq_prev).flatten()
                        else:
                            print("Warning: coeffs_eq_prev is empty, skipping L1 term")

                # L2⊙F项: Δt(1-2γ) L2(U^(1))F(U^(1))
                if self.fitter.has_operator("L2") and self.fitter.has_operator("F"):
                    L2_seg = self.fitter._linear_operators[segment_idx].get("L2", None)
                    u_prev = features @ coeffs_eq_prev  # 转换为解值
                    F_vals = self.fitter.F_func(self.fitter._features[segment_idx], u_prev)
                    rhs[:, eq_idx] += dt * (1 - 2 * gamma) * (L2_seg @  coeffs_eq_prev * F_vals)

                # N项
                if self.fitter.has_operator("N"):
                    u_prev = features @ coeffs_eq_prev  # 转换为解值
                    N_vals = self.fitter.N_func(u_prev)
                    rhs[:, eq_idx] += dt * (1 - gamma) * N_vals
        return rhs

    def get_scheme_info(self) -> Dict[str, Any]:
        """获取IMEX-RK(2,2,2)时间格式信息"""
        return {
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

    def validate_operators(self) -> Dict[str, bool]:
        """验证算子配置是否正确"""
        validation = {
            "L1_exists": self.fitter.has_operator("L1"),
            "L2_exists": self.fitter.has_operator("L2"),
            "N_exists": self.fitter.has_operator("N"),
            "F_exists": self.fitter.has_operator("F"),
            "precompiled": self.fitter._precompiled,
        }

        # 检查IMEX-RK的最小要求
        validation["imex_ready"] = validation["precompiled"] and (
            validation["L1_exists"] or validation["L2_exists"]
        )

        return validation

    def estimate_stable_dt(
        self, u_current: np.ndarray, safety_factor: float = 0.8
    ) -> float:
        """估算稳定的时间步长"""

        if not self.fitter._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        max_eigenvalue = 0.0

        # 估算L1的谱半径
        if self.fitter.has_operator("L1"):
            for segment_idx in range(self.fitter.ns):
                L1_op = self.fitter._linear_operators[segment_idx].get("L1", None)
                if L1_op is not None:
                    try:
                        eigenvals = np.linalg.eigvals(L1_op)
                        max_eigenvalue = max(max_eigenvalue, np.max(np.real(eigenvals)))
                    except:
                        # 如果特征值计算失败，使用保守估计
                        max_eigenvalue = max(max_eigenvalue, np.max(np.abs(L1_op)))

        # 估算L2*F'项（简化处理）
        if self.fitter.has_operator("L2") and self.fitter.has_operator("F"):
            max_eigenvalue = max(max_eigenvalue, 1.0)  # 保守估计

        # 估算N'项（简化处理）
        if self.fitter.has_operator("N"):
            max_eigenvalue = max(max_eigenvalue, 1.0)  # 保守估计

        if max_eigenvalue <= 0:
            return 0.1  # 默认值

        # IMEX-RK(2,2,2)的稳定性常数约为1.5
        C_stability = 1.5
        dt_stable = safety_factor * C_stability / max_eigenvalue

        return dt_stable

    def get_butcher_tableau(self) -> Dict[str, np.ndarray]:
        """获取IMEX-RK(2,2,2) Butcher表"""
        return {
            "A_implicit": self.A_imp,
            "A_explicit": self.A_exp,
            "b_weights": self.b,
            "gamma": self.gamma,
        }
