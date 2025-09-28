"""IMEX First-order (Backward Euler) time integration scheme"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import os
from .base_time_scheme import BaseTimeScheme


class ImexFirstOrder(BaseTimeScheme):
    """IMEX First-order (Backward Euler) time integration scheme implementation"""

    def __init__(self, config):
        super().__init__(config)

        # First-order Backward Euler parameters
        self.gamma = 1.0  # Backward Euler coefficient
        self.order = 1

        # Butcher tableau coefficients (simplified for first-order)
        self.A_imp = np.array([[1.0]])
        self.A_exp = np.array([[0.0]])
        self.b = np.array([1.0])

        # Storage for stage solutions
        self.stage_solutions = {}

        # Maintain current segment-level solution values
        self.U_seg_current = None
        self.U_seg_prev = None

    def time_step(
        self, U_n: np.ndarray, U_seg: List[np.ndarray], dt: float, coeffs_n: np.ndarray = None, current_time: float = 0.0, step: int = 0
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute IMEX First-order (Backward Euler) time step"""
        
        # Store current time for plotting
        self._current_time = current_time

        # Initialize or update current segment-level solution values
        self.U_seg_n = U_seg.copy()

        # Single stage: Solve U^{n+1} directly using Backward Euler
        args = {
            "dt": dt,
            "stage": 1,
            "step": step,
            "U_n_seg": self.U_seg_n
        }
        U_seg_new, coeffs_new = self._solve_imex_stage_U_seg(**args)

        # Convert back to global array
        U_new = self.fitter.segments_to_global(U_seg_new)

        return (
            U_new,
            U_seg_new,
            coeffs_new,
        )

    def _solve_imex_stage_U_seg(
        self,
        **kwargs
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """SolveIMEX一阶Phase - 直接Using段级SolutionvalueOperation"""

        # Prepare阶Number of segments据 - 直接Transfer段级Data
        stage_data = {
            "U_n_seg": kwargs.get("U_n_seg"),
            "dt": kwargs.get("dt"),
            "stage": kwargs.get("stage"),
            "operation": "imex_stage",
            "step": kwargs.get("step"),
        }

        # Usingbase_fitter的fitmethodSolvePhaseCoefficients
        coeffs_stage = self.fitter.fit(**stage_data)

        # 将CoefficientsConvert为Solutionvalue
        U_stage_global, U_stage_segments = self.fitter.construct(
            self.fitter.data, self.fitter._current_model, coeffs_stage
        )

        return U_stage_segments, coeffs_stage

    def build_stage_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """BuildIMEX一阶Phase的段Jacobian matrix
        
        一阶format: [V - Δt*L1 - Δt*L2⊙F(U^n)] β^{n+1} = U^n + Δt*N(U^n)
        """

        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.fitter.dgN

        dt = kwargs.get("dt")
        gamma = self.gamma  # = 1.0 for Backward Euler

        # Get预CompilationOperators和Feature matrix
        L1_operators = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)
        features_list = self.fitter._features[segment_idx]
        features = features_list[0]  # 0 order derivative

        U_n_seg_list = kwargs.get("U_n_seg", [])
        U_n_seg = U_n_seg_list[segment_idx]

        F_n = self.fitter.F_func(features, U_n_seg)
        N_n = self.fitter.N_func(features, U_n_seg) if self.fitter.has_operator("N") else None

        # Processfeatures的Differentformat
        V = features_list[0]  # AllDerivativesinformation都At这Inside 
       
        # BuildSystemMatrix sumRightEndVector
        L_final = np.zeros((ne * n_points, ne * dgN))
        b_final = []

        # ConstructRightEndVector
        kwargs_rhs = {
            "L1_operators": L1_operators,
            "L2_operators": L2_operators,
            "U_n_seg": U_n_seg,
            "F_n": F_n,
            "features": features,
            "N_n": N_n,
            "stage": 1,
            "dt": dt
        }

        rhs = self._build_stage_rhs(segment_idx, **kwargs_rhs)

        # 对EachEquationBuildJacobian matrix
        for eq_idx in range(ne):
            L1 = L1_operators[eq_idx]
            L2 = L2_operators[eq_idx]
            # 行IndexRange
            row_start = eq_idx * n_points
            row_end = (eq_idx + 1) * n_points
            
            # 列IndexRange
            col_start = eq_idx * dgN
            col_end = (eq_idx + 1) * dgN
            
            # Build该Equation的Jacobian matrix: V - Δt*L1 - Δt*L2⊙F
            J_eq = V.copy()  # FromFeature matrixVStart
            
            # 减Go隐式L1Item: -Δt*L1
            if L1 is not None:
                J_eq -= dt * L1
            
            # 减Go隐式L2⊙FItem: -Δt*diag(F)*L2
            if L2 is not None and F_n is not None:
                F_eq = F_n[:, eq_idx]
                # L2⊙FItemLinear化: -Δt * diag(F_eq) @ L2
                L2F_term = dt * np.diag(F_eq) @ L2
                J_eq -= L2F_term
            
            # 将该Equation的Jacobian matrix放入FinalMatrix
            L_final[row_start:row_end, col_start:col_end] = J_eq
            # 添加对应的RightEndItem
            b_final.append(rhs[:, eq_idx])
        
        # 展平RightEndVector
        b_vector = np.concatenate(b_final)
        return L_final, b_vector

    def _build_stage_rhs(self, segment_idx: int, **kwargs) -> np.ndarray:
        """BuildIMEX一阶Phase的RightEndVector
        
        一阶format RHS: U^n + Δt*N(U^n)
        """

        ne = self.config.n_eqs
        dt = kwargs.get("dt")

        # FromkwargsGet预Transfer的Parameter
        U_n_seg = kwargs.get("U_n_seg")
        N_n = kwargs.get("N_n")

        # GetDimensionsinformation
        n_points = U_n_seg.shape[0]

        # InitializeRightEndVector: (n_points, ne)
        rhs = np.zeros((n_points, ne))

        # 一阶format RHS: U^n + Δt*N(U^n)
        # 基础Item: U^n
        rhs[:, :] = U_n_seg
        
        # 添加NonlinearItem: Δt*N(U^n)
        if N_n is not None:
            rhs += dt * N_n

        return rhs

    def get_scheme_info(self) -> Dict[str, Any]:
        """GetIMEX一阶Timeformatinformation"""
        return {
            "method": "IMEX First-order (Backward Euler)",
            "stages": 1,
            "order": 1,
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
        """VerificationOperatorsConfigurationYesNoCorrect"""
        validation = {
            "L1_exists": self.fitter.has_operator("L1"),
            "L2_exists": self.fitter.has_operator("L2"),
            "N_exists": self.fitter.has_operator("N"),
            "F_exists": self.fitter.has_operator("F"),
            "precompiled": self.fitter._precompiled,
        }

        # CheckIMEX的最小Requirement
        validation["imex_ready"] = validation["precompiled"] and (
            validation["L1_exists"] or validation["L2_exists"]
        )

        return validation

    def estimate_stable_dt(
        self, _U_current: np.ndarray, safety_factor: float = 0.8
    ) -> float:
        """估算Stable的Time step size"""

        if not self.fitter._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        max_eigenvalue = 0.0

        # 估算L1的Spectral radius
        if self.fitter.has_operator("L1"):
            for segment_idx in range(self.fitter.ns):
                L1_op = self.fitter._linear_operators[segment_idx].get("L1", None)
                if L1_op is not None:
                    try:
                        eigenvals = np.linalg.eigvals(L1_op)
                        max_eigenvalue = max(max_eigenvalue, np.max(np.real(eigenvals)))
                    except:
                        # IfEigenvalueComputeFail，Using保守Estimation
                        max_eigenvalue = max(max_eigenvalue, np.max(np.abs(L1_op)))

        # 估算L2*F'Item（简化Process）
        if self.fitter.has_operator("L2") and self.fitter.has_operator("F"):
            max_eigenvalue = max(max_eigenvalue, 1.0)  # 保守Estimation

        # 估算N'Item（简化Process）
        if self.fitter.has_operator("N"):
            max_eigenvalue = max(max_eigenvalue, 1.0)  # 保守Estimation

        if max_eigenvalue <= 0:
            return 0.1  # Defaultvalue

        # 一阶format的Stability常数
        C_stability = 2.0
        dt_stable = safety_factor * C_stability / max_eigenvalue

        return dt_stable

    def get_butcher_tableau(self) -> Dict[str, np.ndarray]:
        """GetIMEX一阶format Butcher表"""
        return {
            "A_implicit": self.A_imp,
            "A_explicit": self.A_exp,
            "b_weights": self.b,
            "gamma": self.gamma,
        }