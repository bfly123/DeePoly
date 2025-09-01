"""IMEX-RK(2,2,2) time integration scheme"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import os
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
        self.U_seg_current = (
            None  # List[np.ndarray] - current timestep segment-level solution values
        )
        self.U_seg_prev = (
            None  # List[np.ndarray] - previous timestep segment-level solution values
        )

    def time_step(
        self, U_n: np.ndarray, U_seg: List[np.ndarray], dt: float, coeffs_n: np.ndarray = None, current_time: float = 0.0
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute IMEX-RK(2,2,2) time step"""
        
        # Store current time for plotting
        self._current_time = current_time

        # Initialize or update current segment-level solution values
        self.U_seg_current = U_seg.copy()

        # Stage 1: Solve U^(1)
        U_seg_stage1, coeffs_stage1 = self._solve_imex_stage_U_seg(
            self.U_seg_current, None, dt, stage=1, coeffs_n=coeffs_n
        )
        
        # Plot Stage 1 intermediate solution (disable for performance)
        # self._plot_stage_solution(U_seg_stage1, stage=1, dt=dt)

        # Stage 2: Solve U^(2)
        U_seg_stage2, coeffs_stage2 = self._solve_imex_stage_U_seg(
            self.U_seg_current, U_seg_stage1, dt, stage=2, coeffs_n=coeffs_stage1
        )

        # Final update: Calculate U^{n+1} (according to formula 3.4)
        U_seg_new = self._imex_final_update_U_seg(
            self.U_seg_current,
            U_seg_stage1,
            U_seg_stage2,
            dt,
            coeffs_stage1,
            coeffs_stage2,
        )

        # Update maintained segment-level solution values
        # self.U_seg_prev = self.U_seg_current
        # self.U_seg_current = U_seg_new

        # Convert back to global array
        U_new = self.fitter.segments_to_global(U_seg_new)
        #U_new = self.fitter.segments_to_global(U_seg_stage1)

        return (
            U_new,
            U_seg_new,
            coeffs_stage2,
        )  # Return coefficients from the last stage

    def _solve_imex_stage_U_seg(
        self,
        U_n_seg: List[np.ndarray],
        U_prev_seg: List[np.ndarray],
        dt: float,
        stage: int,
        coeffs_n: np.ndarray = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """求解IMEX-RK阶段 - 直接使用段级解值操作"""

        # 准备阶段数据 - 直接传递段级数据
        stage_data = {
            "U_n_seg": U_n_seg,
            "U_prev_seg": U_prev_seg,
            "U_seg_current": U_n_seg,  # 当前段级解值
            "U_seg_stage1": U_prev_seg,  # 第一阶段结果（对于stage 2）
            "dt": dt,
            "stage": stage,
            "gamma": self.gamma,
            "operation": "imex_stage",
            "coeffs_n": coeffs_n,
        }

        # 使用base_fitter的fit方法求解阶段系数
        coeffs_stage = self.fitter.fit(**stage_data)

        # 将系数转换为解值
        U_stage_global, U_stage_segments = self.fitter.construct(
            self.fitter.data, self.fitter._current_model, coeffs_stage
        )

        return U_stage_segments, coeffs_stage

    def _imex_final_update_U_seg(
        self,
        U_n_seg: List[np.ndarray],
        U_seg_stage1: List[np.ndarray],
        U_seg_stage2: List[np.ndarray],
        dt: float,
        coeffs_stage1: np.ndarray,
        coeffs_stage2: np.ndarray,
    ) -> List[np.ndarray]:
        """IMEX-RK最终更新步骤 - 直接使用段级解值和系数"""
        dt = 0.01

        U_seg_new = []

        for segment_idx in range(self.fitter.ns):
            n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
            U_n_seg_current = U_n_seg[segment_idx]

            # Calculate operator contributions from each stage
            total_contribution = np.zeros_like(U_n_seg_current)

            for stage_idx, (U_seg_stage, coeffs_stage, weight) in enumerate(
                [
                    (U_seg_stage1[segment_idx], coeffs_stage1, self.b[0]),
                    (U_seg_stage2[segment_idx], coeffs_stage2, self.b[1]),
                ]
            ):
                stage_contribution = np.zeros_like(U_n_seg_current)
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

                # N term: N(U^(i)) - nonlinear, directly using U values as input
                if self.fitter.has_operator("N"):
                    N_vals = self.fitter.N_func(
                        self.fitter._features[segment_idx], U_seg_stage
                    )
                    if hasattr(N_vals, "__len__") and len(N_vals) == n_points:
                        stage_contribution += N_vals
                    else:
                        stage_contribution += N_vals

                # L2⊙F term: diag(F) @ L2 @ β^(i) - 统一使用矩阵乘法方式
                if self.fitter.has_operator("L2") and self.fitter.has_operator("F"):
                    L2_seg = self.fitter._linear_operators[segment_idx].get("L2", None)
                    if L2_seg is not None:
                        # 如果L2_seg是3D，取第一个切片
                        if L2_seg.ndim == 3:
                            L2_seg_2d = L2_seg[0]  # 取第一个切片，变成2D
                        else:
                            L2_seg_2d = L2_seg
                            
                        F_vals = self.fitter.F_func(
                            self.fitter._features[segment_idx], U_seg_stage
                        )

                        # 处理F_vals的格式
                        if isinstance(F_vals, list):
                            F_vals = np.array(F_vals[0] if len(F_vals) == 1 else F_vals).T
                        elif F_vals.ndim == 1:
                            F_vals = F_vals.reshape(-1, 1)

                        for eq_idx in range(ne):
                            # 处理不同的系数格式
                            if coeffs_stage.ndim == 3:  # (ns, ne, dgN)
                                beta_seg_stage = coeffs_stage[segment_idx, eq_idx, :]
                            elif coeffs_stage.ndim == 1:  # 展平格式
                                start_idx = (segment_idx * ne + eq_idx) * dgN
                                end_idx = start_idx + dgN
                                beta_seg_stage = coeffs_stage[start_idx:end_idx]
                            else:
                                beta_seg_stage = np.zeros(dgN)
                            
                            # 取当前方程的F值
                            if F_vals.ndim > 1 and F_vals.shape[1] > eq_idx:
                                F_eq = F_vals[:, eq_idx]
                            else:
                                F_eq = F_vals.flatten()
                            
                            # 统一计算方式: diag(F) @ L2 @ β - 与雅可比矩阵一致
                            L2F_contrib = np.diag(F_eq) @ L2_seg_2d @ beta_seg_stage  # (n_points,)
                            
                            # 确保维度兼容
                            if stage_contribution.ndim == 2 and L2F_contrib.ndim == 1:
                                L2F_contrib = L2F_contrib.reshape(-1, 1)
                            elif stage_contribution.ndim == 1 and L2F_contrib.ndim == 2:
                                L2F_contrib = L2F_contrib.flatten()
                                
                            stage_contribution += L2F_contrib

                # Weighted accumulation
                total_contribution += weight * stage_contribution

            # Final update: U^{n+1} = U^n + Δt * total_contribution
            U_seg_new.append(U_n_seg_current + dt * total_contribution)

        return U_seg_new

    def build_stage_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建IMEX-RK阶段的段雅可比矩阵
        
        根据Doc/time_scheme_program.md的公式：
        阶段1: [V - γΔt*L1 - γΔt*L2⊙F(U^n)] β^(1) = U^n + γΔt*N(U^n)
        阶段2: [V - γΔt*L1 - γΔt*L2⊙F(U^(1))] β^(2) = RHS
        """

        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.fitter.dgN

        stage = kwargs.get("stage", 1)
        dt = kwargs.get("dt", 0.01)
        gamma = kwargs.get("gamma", self.gamma)
        #gamma = 1
        dt = 0.01

        # 获取预编译算子和特征矩阵
        L1_operators = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)
        features_list = self.fitter._features[segment_idx]  # 可能是列表格式
        
        # 处理features的不同格式
        if isinstance(features_list, list):
            features = features_list[0]  # 取第一个元素作为特征矩阵
        else:
            features = features_list
        
        # 检查L1算子实际结构并适配
        
        if L1_operators is not None:
            if L1_operators.ndim == 3:
                # 如果是3D，取第一个切片作为通用L1算子
                L1_2d = L1_operators[0]  # (n_points, dgN)
            else:
                L1_2d = L1_operators
        else:
            L1_2d = None
            
        if L2_operators is not None:
            if L2_operators.ndim == 3:
                L2_2d = L2_operators[0]  # (n_points, dgN)
            else:
                L2_2d = L2_operators
        else:
            L2_2d = None

        # 构建系统矩阵和右端向量
        L_final = np.zeros((ne * n_points, ne * dgN))
        b_final = []

        # 构造右端向量 - 移除重复的stage参数
        kwargs_rhs = kwargs.copy()
        kwargs_rhs['stage'] = stage  # 确保使用正确的stage值
        rhs = self._build_stage_rhs(segment_idx, **kwargs_rhs)

        # 对每个方程构建雅可比矩阵
        for eq_idx in range(ne):
            # 行索引范围
            row_start = eq_idx * n_points
            row_end = (eq_idx + 1) * n_points
            
            # 列索引范围
            col_start = eq_idx * dgN
            col_end = (eq_idx + 1) * dgN
            
            # 构建该方程的雅可比矩阵: V - γΔt*L1 - γΔt*L2⊙F
            J_eq = features.copy()  # 从特征矩阵V开始
            
            # 减去隐式L1项: -γΔt*L1
            if L1_2d is not None:
              if stage == 1:
                J_eq -= gamma * dt * L1_2d
              elif stage == 2:
                J_eq -= (1-gamma) * dt * L1_2d
            
            # 减去隐式L2⊙F项: -γΔt*diag(F)*L2 (统一使用逐点相乘方式的线性化形式)
            if L2_2d is not None and self.fitter.has_operator("F"):
                # 获取当前U值
                U_n_seg_list = kwargs.get("U_n_seg", [])
                U_prev_seg_list = kwargs.get("U_prev_seg", [])
                
                # 根据阶段选择U值
                if stage == 1:
                    U_seg_for_F = U_n_seg_list[segment_idx] if U_n_seg_list else None
                else:
                    U_seg_for_F = U_prev_seg_list[segment_idx] if U_prev_seg_list else None
                
                if U_seg_for_F is not None:
                    # 计算F值: (n_points, ne)
                    F_vals = self.fitter.F_func(features, U_seg_for_F)
                    
                    # Convert F_vals to numpy array if it's a list
                    if isinstance(F_vals, list):
                        F_vals = np.array(F_vals[0] if len(F_vals) == 1 else F_vals).T  # Transpose to get (n_points, ne)
                    elif F_vals.ndim == 1:
                        F_vals = F_vals.reshape(-1, 1)
                    
                    # 取当前方程的F值
                    if F_vals.ndim > 1 and F_vals.shape[1] > eq_idx:
                        F_eq = F_vals[:, eq_idx]  # (n_points,)
                    else:
                        F_eq = F_vals.flatten()
                    
                    # 统一的L2⊙F项线性化: -γΔt * diag(F_eq) @ L2_2d
                    # 这对应于 (L2@β) ⊙ F 的雅可比矩阵形式
                    L2F_term = gamma * dt * np.diag(F_eq) @ L2_2d
                    J_eq -= L2F_term
            
            # 将该方程的雅可比矩阵放入最终矩阵
            L_final[row_start:row_end, col_start:col_end] = J_eq
            
            # 添加对应的右端项
            b_final.append(rhs[:, eq_idx])

        # 展平右端向量
        b_vector = np.concatenate(b_final)

        return L_final, b_vector

    def _build_stage_rhs(self, segment_idx: int, **kwargs) -> np.ndarray:
        """构建IMEX-RK阶段的右端向量
        
        根据Doc/time_scheme_program.md的公式：
        阶段1 RHS: U^n + γΔt*N(U^n)
        阶段2 RHS: U^n + Δt(1-2γ)[L1 + L2⊙F(U^(1))]β^(1) + Δt(1-γ)N(U^(1))
        """

        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dt = kwargs.get("dt", 0.01)
        dt = 0.01
        gamma = kwargs.get("gamma", self.gamma)
        stage = kwargs.get("stage", 1)

        # 获取算子并处理3D情况
        L1_operators = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)
        features_list = self.fitter._features[segment_idx]
        
        # 处理features的不同格式
        if isinstance(features_list, list):
            features = features_list[0]  # 取第一个元素作为特征矩阵
        else:
            features = features_list

        # 适配L1, L2算子维度
        if L1_operators is not None and L1_operators.ndim == 3:
            L1_2d = L1_operators[0]  # (n_points, dgN)
        else:
            L1_2d = L1_operators
            
        if L2_operators is not None and L2_operators.ndim == 3:
            L2_2d = L2_operators[0]  # (n_points, dgN) 
        else:
            L2_2d = L2_operators

        # 初始化右端向量: (n_points, ne)
        rhs = np.zeros((n_points, ne))

        if stage == 1:
            # 阶段1 RHS: U^n + γΔt*N(U^n)
            U_n_seg_list = kwargs.get("U_n_seg", [])
            
            if U_n_seg_list and len(U_n_seg_list) > segment_idx:
                U_n_seg = U_n_seg_list[segment_idx]  # (n_points, ne)
                
                # 基础项: U^n
                rhs[:, :] = U_n_seg
                
                # 添加非线性项: γΔt*N(U^n)
                if self.fitter.has_operator("N"):
                    N_vals = self.fitter.N_func(features, U_n_seg)  # (n_points, ne)
                    rhs += gamma * dt * N_vals

        elif stage == 2:
            # 阶段2 RHS: U^n + Δt(1-2γ)[L1 + L2⊙F(U^(1))]β^(1) + Δt(1-γ)N(U^(1))
            U_n_seg_list = kwargs.get("U_n_seg", [])
            U_seg_stage1_list = kwargs.get("U_seg_stage1", [])
            coeffs_stage1 = kwargs.get("coeffs_n", None)  # 阶段1的系数
            
            if U_n_seg_list and len(U_n_seg_list) > segment_idx:
                U_n_seg = U_n_seg_list[segment_idx]
                
                # 基础项: U^n
                rhs[:, :] = U_n_seg
                
                # 如果有阶段1的数据，计算显式项
                if (U_seg_stage1_list and len(U_seg_stage1_list) > segment_idx and 
                    coeffs_stage1 is not None):
                    
                    U_seg_stage1 = U_seg_stage1_list[segment_idx]
                    
                    # 对每个方程处理显式项
                    for eq_idx in range(ne):
                        # 提取阶段1系数 β^(1)
                        if coeffs_stage1.ndim == 3:  # (ns, ne, dgN)
                            beta_1 = coeffs_stage1[segment_idx, eq_idx, :]
                        elif coeffs_stage1.ndim == 1:  # 展平格式
                            start_idx = (segment_idx * ne + eq_idx) * self.fitter.dgN
                            end_idx = start_idx + self.fitter.dgN
                            beta_1 = coeffs_stage1[start_idx:end_idx]
                        else:
                            beta_1 = np.zeros(self.fitter.dgN)
                        
                        # L1项: Δt(1-2γ)*L1*β^(1)
                        if L1_2d is not None:
                            L1_contrib = L1_2d @ beta_1  # (n_points,)
                            rhs[:, eq_idx] += dt * (1-2*gamma) * L1_contrib
                        
                        # L2⊙F项: Δt(1-2γ)*diag(F(U^(1)))*L2*β^(1) - 统一矩阵乘法方式
                        if L2_2d is not None and self.fitter.has_operator("F"):
                            F_vals = self.fitter.F_func(features, U_seg_stage1)  # (n_points, ne)
                            
                            # Convert F_vals to numpy array if it's a list
                            if isinstance(F_vals, list):
                                F_vals = np.array(F_vals[0] if len(F_vals) == 1 else F_vals).T  # Transpose to get (n_points, ne)
                            elif F_vals.ndim == 1:
                                F_vals = F_vals.reshape(-1, 1)
                            
                            # 取当前方程的F值
                            if F_vals.ndim > 1 and F_vals.shape[1] > eq_idx:
                                F_eq = F_vals[:, eq_idx]
                            else:
                                F_eq = F_vals.flatten()
                            
                            # 统一计算方式: diag(F) @ L2 @ β - 与雅可比矩阵一致
                            L2F_contrib = np.diag(F_eq) @ L2_2d @ beta_1  # (n_points,)
                            rhs[:, eq_idx] += dt * (1-2*gamma) * L2F_contrib
                    
                    # N项: Δt(1-γ)*N(U^(1)) - 对所有方程一起计算
                    if self.fitter.has_operator("N"):
                        N_vals = self.fitter.N_func(features, U_seg_stage1)  # (n_points, ne)
                        rhs += dt * (1-gamma) * N_vals

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
            "equation_form": "dU/dt = L1(U) + N(U) + L2(U)*F(U)",
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
        self, U_current: np.ndarray, safety_factor: float = 0.8
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

    def _debug_L2F_term(self, segment_idx: int, U_seg: np.ndarray, stage: int):
        """Debug L2*F term to check if it's working correctly"""
        print(f"\n=== Debugging L2⊙F Term (Stage {stage}, Segment {segment_idx}) ===")
        
        # Get L2 operator
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)
        if L2_operators is not None:
            if L2_operators.ndim == 3:
                L2_seg = L2_operators[0]  # (n_points, dgN)
            else:
                L2_seg = L2_operators
            print(f"L2 operator shape: {L2_seg.shape}")
            print(f"L2 operator range: [{L2_seg.min():.6e}, {L2_seg.max():.6e}]")
            print(f"L2 operator norm: {np.linalg.norm(L2_seg):.6e}")
        else:
            print("L2 operator not found!")
            return
        
        # Get features
        features_list = self.fitter._features[segment_idx]
        if isinstance(features_list, list):
            features = features_list[0]
        else:
            features = features_list
        print(f"Features shape: {features.shape}")
        print(f"Features range: [{features.min():.6e}, {features.max():.6e}]")
        
        # Check if L2 and features are the same
        if np.allclose(L2_seg, features):
            print("✓ L2 operator matches features matrix (as expected)")
        else:
            print("✗ L2 operator does NOT match features matrix!")
            diff_norm = np.linalg.norm(L2_seg - features)
            print(f"  Difference norm: {diff_norm:.6e}")
        
        # Calculate F values
        if self.fitter.has_operator("F"):
            print(f"U_seg shape: {U_seg.shape}")
            print(f"U_seg range: [{U_seg.min():.6e}, {U_seg.max():.6e}]")
            
            F_vals = self.fitter.F_func(features, U_seg)
            
            # Convert F_vals to numpy array if it's a list
            if isinstance(F_vals, list):
                F_vals = np.array(F_vals[0] if len(F_vals) == 1 else F_vals).T  # Transpose to get (n_points, ne)
            elif F_vals.ndim == 1:
                F_vals = F_vals.reshape(-1, 1)
            
            print(f"F function values shape: {F_vals.shape}")
            print(f"F function values range: [{F_vals.min():.6e}, {F_vals.max():.6e}]")
            
            # Expected F = 5 - 5*u^2
            expected_F = 5 - 5 * U_seg**2
            print(f"Expected F = 5-5*U^2 range: [{expected_F.min():.6e}, {expected_F.max():.6e}]")
            
            if np.allclose(F_vals, expected_F.flatten() if F_vals.ndim == 1 else expected_F):
                print("✓ F function matches expected formula 5-5*U^2")
            else:
                diff_norm = np.linalg.norm(F_vals - expected_F.flatten() if F_vals.ndim == 1 else expected_F)
                print(f"✗ F function differs from expected! Diff norm: {diff_norm:.6e}")
            
            # Test L2⊙F operation
            ne = self.config.n_eqs
            dgN = self.fitter.dgN
            for eq_idx in range(ne):
                # Get some test coefficients (use ones for testing)
                beta_test = np.ones(dgN)
                
                # Calculate L2 @ beta
                L2_beta = L2_seg @ beta_test
                print(f"L2 @ β (test) range: [{L2_beta.min():.6e}, {L2_beta.max():.6e}]")
                
                # Get F for current equation
                if F_vals.ndim > 1 and F_vals.shape[1] > eq_idx:
                    F_eq = F_vals[:, eq_idx]
                else:
                    F_eq = F_vals.flatten()
                
                # Calculate L2⊙F contribution
                L2F_contrib = L2_beta * F_eq
                print(f"L2⊙F contribution range: [{L2F_contrib.min():.6e}, {L2F_contrib.max():.6e}]")
                print(f"L2⊙F contribution norm: {np.linalg.norm(L2F_contrib):.6e}")
                
                # Compare with pure L2 contribution
                L2_contrib = L2_beta
                print(f"Pure L2 contribution norm: {np.linalg.norm(L2_contrib):.6e}")
                
                ratio = np.linalg.norm(L2F_contrib) / (np.linalg.norm(L2_contrib) + 1e-12)
                print(f"L2⊙F/L2 ratio: {ratio:.6f}")
                
        else:
            print("F operator not found!")
        
        print("=" * 60)

    def _plot_stage_solution(self, U_seg_stage: List[np.ndarray], stage: int, dt: float):
        """Plot intermediate stage solution for debugging and analysis"""
        try:
            # Debug L2*F term for the first segment
            if stage == 1 and len(U_seg_stage) > 0:
                self._debug_L2F_term(0, U_seg_stage[0], stage)
            
            # Convert segment solution to global array for plotting
            U_stage_global = self.fitter.segments_to_global(U_seg_stage)
            
            # Get coordinates for plotting
            if hasattr(self.fitter, 'data') and 'x' in self.fitter.data:
                x_coords = self.fitter.data['x']
            elif hasattr(self.fitter, 'data') and 'x_segments_norm' in self.fitter.data:
                # Combine all segment coordinates
                x_coords = []
                for seg_idx in range(len(self.fitter.data['x_segments_norm'])):
                    x_coords.append(self.fitter.data['x_segments_norm'][seg_idx])
                x_coords = np.vstack(x_coords)
            else:
                print(f"Warning: Cannot find coordinates for stage {stage} plotting")
                return
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            if x_coords.shape[1] == 1:  # 1D case
                x_flat = x_coords.flatten()
                
                # Sort for proper line plotting
                sort_idx = np.argsort(x_flat)
                x_sorted = x_flat[sort_idx]
                
                # Plot each equation
                for eq_idx in range(U_stage_global.shape[1]):
                    U_sorted = U_stage_global[sort_idx, eq_idx]
                    plt.plot(x_sorted, U_sorted, 'o-', 
                            label=f'U_{eq_idx} Stage {stage}', 
                            linewidth=2, markersize=4, alpha=0.8)
                
                plt.xlabel('x', fontsize=12)
                plt.ylabel('U(x)', fontsize=12)
                plt.title(f'IMEX-RK Stage {stage} Solution (dt={dt:.6f})', 
                         fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
            elif x_coords.shape[1] == 2:  # 2D case
                # Create scatter plot for 2D
                for eq_idx in range(U_stage_global.shape[1]):
                    plt.scatter(x_coords[:, 0], x_coords[:, 1], 
                               c=U_stage_global[:, eq_idx], 
                               cmap='RdYlBu_r', s=20, alpha=0.8)
                    plt.colorbar(label=f'U_{eq_idx}')
                
                plt.xlabel('x', fontsize=12)
                plt.ylabel('y', fontsize=12)
                plt.title(f'IMEX-RK Stage {stage} Solution (dt={dt:.6f})', 
                         fontsize=14, fontweight='bold')
                plt.axis('equal')
            
            # Save the plot
            if hasattr(self.fitter, 'config') and hasattr(self.fitter.config, 'case_dir'):
                results_dir = os.path.join(self.fitter.config.case_dir, 'results')
            else:
                results_dir = 'results'
            
            os.makedirs(results_dir, exist_ok=True)
            
            # Get current time step for filename
            current_time = getattr(self, '_current_time', 0.0)
            filename = f'imex_stage_{stage}_t_{current_time:.6f}_dt_{dt:.6f}.png'
            filepath = os.path.join(results_dir, filename)
            
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()  # Close to free memory
            
            print(f"  Stage {stage} solution plot saved: {filepath}")
            
            # Print solution statistics
            U_min = U_stage_global.min()
            U_max = U_stage_global.max()
            U_norm = np.linalg.norm(U_stage_global)
            print(f"  Stage {stage} solution stats: min={U_min:.6e}, max={U_max:.6e}, norm={U_norm:.6e}")
            
        except Exception as e:
            print(f"Warning: Failed to plot stage {stage} solution: {str(e)}")
            # Don't raise the error - plotting is optional
