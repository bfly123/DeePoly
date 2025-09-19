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
        self.order = 2

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
        self, U_n: np.ndarray, U_seg: List[np.ndarray], dt: float, coeffs_n: np.ndarray = None, current_time: float = 0.0,step: int = 0
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute IMEX-RK(2,2,2) time step"""
        
        # Store current time for plotting
        self._current_time = current_time

        # Initialize or update current segment-level solution values
        self.U_seg_n = U_seg.copy()

        # Stage 1: Solve U^(1)
        args={
          "dt": dt,
          "stage": 1,
          "step": step,
          "U_n_seg": self.U_seg_n
        #  "coeffs_n": coeffs_n,
        }
        U_seg_stage1, coeffs_stage1 = self._solve_imex_stage_U_seg(
          **args
        )

        args={
          "dt": dt,
          "stage": 2,
          "step": step,
          "U_n_seg": self.U_seg_n,
          "U_1_seg": U_seg_stage1,
          "coeffs_1": coeffs_stage1,
        }

        # Plot Stage 1 intermediate solution (disable for performance)
        # self._plot_stage_solution(U_seg_stage1, stage=1, dt=dt)

        # Stage 2: Solve U^(2)
        U_seg_stage2, coeffs_stage2 = self._solve_imex_stage_U_seg(
          **args
        )

        # Final update: Calculate U^{n+1} (according to formula 3.4)
        args={
          "U_n_seg": self.U_seg_n,
          "U_1_seg": U_seg_stage1,
          "U_2_seg": U_seg_stage2,
          "dt": dt,
          "coeffs_1": coeffs_stage1,
          "coeffs_2": coeffs_stage2,
        }
        U_seg_new = self._imex_final_update_U_seg(
            **args
        )

        # Update maintained segment-level solution values
        # self.U_seg_prev = self.U_seg_current
        # self.U_seg_current = U_seg_new

        # Convert back to global array
        U_new = self.fitter.segments_to_global(U_seg_new)

        return (
            U_new,
            U_seg_new,
            coeffs_stage2,
        )

    def _solve_imex_stage_U_seg(
        self,
        **kwargs
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """求解IMEX-RK阶段 - 直接使用段级解值操作"""

        # 准备阶段数据 - 直接传递段级数据
        stage_data = {
            "U_n_seg": kwargs.get("U_n_seg"),
            "U_1_seg": kwargs.get("U_1_seg",None),
            "dt": kwargs.get("dt"),
            "stage": kwargs.get("stage"),
            "operation": "imex_stage",
            "coeffs_1": kwargs.get("coeffs_1",None),
            "step": kwargs.get("step"),
        }

        # 使用base_fitter的fit方法求解阶段系数
        coeffs_stage = self.fitter.fit(**stage_data)

        # 将系数转换为解值
        U_stage_global, U_stage_segments = self.fitter.construct(
            self.fitter.data, self.fitter._current_model, coeffs_stage
        )

        # 计算并存储算子值K用于最终更新
        K_stage_segments = self._compute_stage_operators(
            U_stage_segments, coeffs_stage, kwargs
        )

        # 存储算子值到实例变量
        stage_num = kwargs.get("stage")
        setattr(self, f'_K_stage_{stage_num}', K_stage_segments)

        # 预计算stage2所需的显式项并存储 (like K)
        if stage_num == 1:
            explicit_stage1_segments = self._compute_explicit_stage1_terms(
                U_stage_segments, coeffs_stage
            )
            setattr(self, '_explicit_stage1_terms', explicit_stage1_segments)

        return U_stage_segments, coeffs_stage

    def _compute_stage_operators(
        self, 
        U_stage_segments: List[np.ndarray], 
        coeffs_stage: np.ndarray, 
        kwargs: dict
    ) -> List[np.ndarray]:
        """计算阶段算子值 K = L1(U) + L2(U)⊙F(U) + N(U)"""
        
        K_segments = []
        stage = kwargs.get("stage")
        U_n_seg = kwargs.get("U_n_seg", [])
        
        for segment_idx in range(self.fitter.ns):
            U_seg = U_stage_segments[segment_idx]
            K_seg = np.zeros_like(U_seg)
            
            # 获取算子和特征
            L1_ops = self.fitter._linear_operators[segment_idx].get("L1", None)
            L2_ops = self.fitter._linear_operators[segment_idx].get("L2", None)
            features = self.fitter._features[segment_idx][0]
            
            # 预计算F值（一次计算所有方程）
            F_vals = None
            if L2_ops is not None and self.fitter.has_operator("F"):
                F_vals = self.fitter.F_func(features, U_seg)
            
            # 计算各方程的算子贡献
            for eq_idx in range(self.config.n_eqs):
                beta = coeffs_stage[segment_idx, eq_idx, :]
                
                # L1贡献
                if L1_ops is not None:
                    L1_contrib = L1_ops[eq_idx] @ beta
                    K_seg[:, eq_idx] += L1_contrib
                
                # L2⊙F贡献（使用预计算的F值）
                if L2_ops is not None and F_vals is not None:
                    L2_contrib = L2_ops[eq_idx] @ beta
                    K_seg[:, eq_idx] += L2_contrib * F_vals[:, eq_idx]
            
            # N贡献
            if self.fitter.has_operator("N"):
                # 根据IMEX-RK公式，阶段1使用U^n，阶段2使用U^(1)
                if stage == 1:
                    N_input = U_n_seg[segment_idx]
                else:
                    # 阶段2应该使用U^(1)，从kwargs中获取
                    U_1_seg = kwargs.get("U_1_seg", [])
                    N_input = U_1_seg[segment_idx] if U_1_seg else U_seg
                N_vals = self.fitter.N_func(features, N_input)
                K_seg += N_vals
            
            K_segments.append(K_seg)
        
        return K_segments

    def _compute_explicit_stage1_terms(
        self,
        U_stage1_segments: List[np.ndarray],
        coeffs_stage1: np.ndarray
    ) -> List[np.ndarray]:
        """预计算stage1的显式项 [L1 + L2⊙F(U^(1))]β^(1) 用于stage2的RHS"""

        explicit_segments = []

        for segment_idx in range(self.fitter.ns):
            U_seg = U_stage1_segments[segment_idx]
            n_points = U_seg.shape[0]
            ne = self.config.n_eqs

            # 获取算子和特征
            L1_ops = self.fitter._linear_operators[segment_idx].get("L1", None)
            L2_ops = self.fitter._linear_operators[segment_idx].get("L2", None)
            features = self.fitter._features[segment_idx][0]

            # 预计算F(U^(1))
            F_vals = None
            if L2_ops is not None and self.fitter.has_operator("F"):
                F_vals = self.fitter.F_func(features, U_seg)

            # 计算显式项
            explicit_seg = np.zeros((n_points, ne))

            for eq_idx in range(ne):
                beta_1 = coeffs_stage1[segment_idx, eq_idx, :]

                # L1*β^(1)贡献
                if L1_ops is not None:
                    L1_contrib = L1_ops[eq_idx] @ beta_1
                    explicit_seg[:, eq_idx] += L1_contrib

                # L2*β^(1)*F(U^(1))贡献
                if L2_ops is not None and F_vals is not None:
                    L2_contrib = L2_ops[eq_idx] @ beta_1
                    explicit_seg[:, eq_idx] += L2_contrib * F_vals[:, eq_idx]

            explicit_segments.append(explicit_seg)

        return explicit_segments

    def _imex_final_update_U_seg(
        self,
        U_n_seg: List[np.ndarray],
        dt: float,
        **_kwargs
    ) -> List[np.ndarray]:
        """IMEX-RK最终更新步骤 - 直接使用预计算的算子值K"""
        
        # 检查是否有预计算的算子值
        if not hasattr(self, '_K_stage_1') or not hasattr(self, '_K_stage_2'):
            raise RuntimeError("Stage operator values not computed. This is a programming error.")
        
        U_seg_new = []
        
        # 对每个段进行最终更新
        for segment_idx in range(self.fitter.ns):
            U_n_current = U_n_seg[segment_idx]
            K1 = -self._K_stage_1[segment_idx]
            K2 = -self._K_stage_2[segment_idx]
            
            # 根据IMEX-RK(2,2,2)公式: U^{n+1} = U^n + dt/2 * [K^{(1)} + K^{(2)}]
            U_new_seg = U_n_current + dt * 0.5 * (K1 + K2)
            U_seg_new.append(U_new_seg)
        
        return U_seg_new

    def build_stage_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建IMEX-RK阶段的段雅可比矩阵
        
        根据Doc/time_scheme_program.md的公式：
        阶段1: [V + γΔt*L1 + γΔt*L2⊙F(U^n)] β^(1) = U^n - γΔt*N(U^n)
        阶段2: [V + γΔt*L1 + γΔt*L2⊙F(U^(1))] β^(2) = RHS
        """

        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.fitter.dgN

        stage = kwargs.get("stage")
        dt = kwargs.get("dt")
        gamma = self.gamma
        #dt = 0.01

        # 获取预编译算子和特征矩阵
        L1_operators = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)
        features_list = self.fitter._features[segment_idx]
        features = features_list[0] #0 order derivative

        U_n_seg_list = kwargs.get("U_n_seg", [])
        U_n_seg = U_n_seg_list[segment_idx]
        U_prev_seg_list = kwargs.get("U_prev_seg", [])
        U_prev_seg = U_prev_seg_list[segment_idx] if U_prev_seg_list else None

        F_n = self.fitter.F_func(features, U_n_seg)
        F_prev = self.fitter.F_func(features, U_prev_seg) if U_prev_seg is not None else None

        N_n = self.fitter.N_func(features, U_n_seg) if self.fitter.has_operator("N") else None
        N_prev = self.fitter.N_func(features, U_prev_seg) if U_prev_seg is not None and self.fitter.has_operator("N") else None

        beta = kwargs.get("coeffs_n", None)
        beta_prev = beta[segment_idx, :, :] if beta is not None else None

        # 处理features的不同格式
        V = features_list[0]  #所有导数信息都在这里 
       
        # 构建系统矩阵和右端向量
        L_final = np.zeros((ne * n_points, ne * dgN))
        b_final = []

        # 构造右端向量 - 移除重复的stage参数
        kwargs_rhs = {
          "L1_operators": L1_operators,
          "L2_operators": L2_operators,
          "U_n_seg": U_n_seg,
          "U_prev_seg": U_prev_seg,
          "F_n": F_n,
          "F_prev": F_prev,
          "features": features,
          "N_n": N_n,
          "N_prev": N_prev,
          "beta_prev": beta_prev,
          "stage": stage,
          "dt": dt
        }

        rhs = self._build_stage_rhs(segment_idx, **kwargs_rhs)

        # 对每个方程构建雅可比矩阵
        for eq_idx in range(ne):
            L1  = L1_operators[eq_idx]
            L2  = L2_operators[eq_idx]
            # 行索引范围
            row_start = eq_idx * n_points
            row_end = (eq_idx + 1) * n_points
            
            # 列索引范围
            col_start = eq_idx * dgN
            col_end = (eq_idx + 1) * dgN
            
            # 构建该方程的雅可比矩阵: V - γΔt*L1 - γΔt*L2⊙F
            J_eq = V.copy()  # 从特征矩阵V开始
            
            # 减去隐式L1项: +γΔt*L1
            if L1 is not None:
                J_eq += gamma * dt * L1
            
            # 减去隐式L2⊙F项: +γΔt*diag(F)*L2
            if L2 is not None:
                if stage == 1:
                    F_vals = F_n
                elif stage == 2:
                    F_vals = F_prev
                else:
                    raise ValueError(f"Invalid stage: {stage}")
                
                # 只有当F_vals不为None时才进行计算
                if F_vals is not None:
                    F_eq = F_vals[:, eq_idx]
                    # L2⊙F项线性化: -γΔt * diag(F_eq) @ L2
                    L2F_term = gamma * dt * np.diag(F_eq) @ L2
                    J_eq += L2F_term
            # 将该方程的雅可比矩阵放入最终矩阵
            L_final[row_start:row_end, col_start:col_end] = J_eq
            # 添加对应的右端项
            b_final.append(rhs[:, eq_idx])
        # 展平右端向量
        b_vector = np.concatenate(b_final)
        return L_final, b_vector

    def _build_stage_rhs(self, _segment_idx: int, **kwargs) -> np.ndarray:
        """构建IMEX-RK阶段的右端向量
        
        根据Doc/time_scheme_program.md的公式：
        阶段1 RHS: U^n - γΔt*N(U^n)
        阶段2 RHS: U^n - Δt(1-2γ)[L1 + L2⊙F(U^(1))]β^(1) + Δt(1-γ)N(U^(1))
        """

        ne = self.config.n_eqs
        dt = kwargs.get("dt")
        gamma = self.gamma
        stage = kwargs.get("stage")

        # 从kwargs获取预传递的参数
        U_n_seg = kwargs.get("U_n_seg")
        N_n = kwargs.get("N_n")
        N_prev = kwargs.get("N_prev")

        # 获取维度信息
        n_points = U_n_seg.shape[0]

        # 初始化右端向量: (n_points, ne)
        rhs = np.zeros((n_points, ne))

        if stage == 1:
            # 阶段1 RHS: U^n - γΔt*N(U^n)
            # 基础项: U^n
            rhs[:, :] = U_n_seg
            
            # 添加非线性项: γΔt*N(U^n)
            if N_n is not None:
                rhs -= gamma * dt * N_n

        elif stage == 2:
            # 阶段2 RHS: U^n + Δt(1-2γ)[L1 + L2⊙F(U^(1))]β^(1) + Δt(1-γ)N(U^(1))
            # 基础项: U^n
            rhs[:, :] = U_n_seg

            # 添加预计算的显式项: Δt(1-2γ)[L1 + L2⊙F(U^(1))]β^(1)
            if hasattr(self, '_explicit_stage1_terms') and self._explicit_stage1_terms:
                explicit_seg = self._explicit_stage1_terms[_segment_idx]
                rhs -= dt * (1 - 2*gamma) * explicit_seg

            # 添加显式非线性项: Δt(1-γ)*N(U^(1))
            if N_prev is not None:
                rhs -= dt * (1-gamma) * N_prev

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
        self, _U_current: np.ndarray, safety_factor: float = 0.8
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
        
        # Get L2 operator - 3D格式取第一个切片
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)
        L2_seg = L2_operators[0] if L2_operators is not None else None
        
        if L2_seg is not None:
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
            
            # F_func返回标准格式 (n_points, ne)
            
            print(f"F function values shape: {F_vals.shape}")
            print(f"F function values range: [{F_vals.min():.6e}, {F_vals.max():.6e}]")
            
            # Expected F = 5 - 5*u^2
            expected_F = 5 - 5 * U_seg**2
            print(f"Expected F = 5-5*U^2 range: [{expected_F.min():.6e}, {expected_F.max():.6e}]")
            
            if np.allclose(F_vals, expected_F):
                print("✓ F function matches expected formula 5-5*U^2")
            else:
                diff_norm = np.linalg.norm(F_vals - expected_F)
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
                
                # Get F for current equation - 标准格式 (n_points, ne)
                F_eq = F_vals[:, eq_idx]
                
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
