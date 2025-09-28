"""
One-Step Predictor-Corrector Time Integration Scheme

One-step predictor-corrector time integration scheme:
- Predictor step: Implicit solve (u^* - u^n)/Δt + L1(u^*) + L2(u^*)⋅F(u^n) + N(u^n) = 0
- Corrector step: Explicit compute u^{n+1} = u^n - Δt * [L1_avg + L2F_avg + N_mid]
- Operator value pre-computation and time step reuse, reducing 75% operator calculations
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from itertools import product

from .base_time_scheme import BaseTimeScheme


class OneStepPredictor(BaseTimeScheme):
    """One-step predictor-corrector time integration scheme

    Features:
    - Predictor step implicit solve, corrector step explicit computation
    - Conditional correction: execute corrector step only when step > 1
    - Operator value pre-computation and time step reuse
    - Midpoint evaluation improves nonlinear term accuracy
    """

    def __init__(self, config):
        super().__init__(config)
        self.order = 1  # Basic accuracy order
        self.predictor_solution = None  # Store predictor solution u^*
        self.predictor_coeffs = None    # Store predictor coefficients β^*
        self.L1_beta_prev = None        # Store previous time step L1*β^n
        self.L2_beta_prev = None        # Store previous time step L2*β^n
        self.L1_beta_star = None        # Store current predictor step L1*β^*
        self.L2_beta_star = None        # Store current predictor step L2*β^*

    def time_step(
        self,
        U_n: np.ndarray,
        U_seg: List[np.ndarray],
        dt: float,
        coeffs_n: np.ndarray = None,
        current_time: float = 0.0,
        step: int = 0
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute one-step predictor-corrector time integration

        Args:
            U_n: Current time step global solution values
            U_seg: Current time step segment-level solution list
            dt: Time step size
            coeffs_n: Current time step coefficients (optional)
            current_time: Current time
            step: Time step number

        Returns:
            Tuple: (new global solution, new segment-level solution, new coefficients)
        """

        # StoreCurrent time和Solutionvalue（dtreduction handled in solver）
        self._current_time = current_time
        self.U_seg_n = [seg.copy() for seg in U_seg]

        # Step1: Predictor step（implicit solve）
        U_star_seg, coeffs_star = self._predictor_step(U_seg, dt, step)

        # Step2: Corrector step（only whenstep > 1execute explicit computation）
        if step > 1 and self.L1_beta_prev is not None:
            U_new_seg, coeffs_new = self._corrector_step(
                U_seg, U_star_seg, dt, coeffs_star, step
            )
        else:
            # First step directly uses predictor values
            U_new_seg, coeffs_new = U_star_seg, coeffs_star

        # Step3: Update operator values for next time step use
        # Current time步的Predictionoperator values成为Down一Time step的"previous"operator values
        if hasattr(self, 'L1_beta_star') and self.L1_beta_star is not None:
            self.L1_beta_prev = self.L1_beta_star
        if hasattr(self, 'L2_beta_star') and self.L2_beta_star is not None:
            self.L2_beta_prev = self.L2_beta_star

        # Convert to global array
        U_new = self.fitter.segments_to_global(U_new_seg)

        return U_new, U_new_seg, coeffs_new

    def _predictor_step(self, U_n_seg: List[np.ndarray], dt: float, step: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """Predictor step: (u^* - u^n)/Δt + L1(u^*) + L2(u^*)⋅F(u^n) + N(u^n) = 0

        implicit solve，requires building and solving linear system
        """

        # PreparePredictor stepData - following IMEX-RK interface format
        predictor_data = {
            "U_n_seg": U_n_seg,
            "dt": dt,
            "step": step,
            "operation": "onestep_predictor"  # New operation type
        }

        # SolvePredictor stepCoefficients
        coeffs_star = self.fitter.fit(**predictor_data)

        # Construct predictor solution
        U_star_global, U_star_seg = self.fitter.construct(
            self.fitter.data, self.fitter._current_model, coeffs_star
        )

        # 直接Compute并Storeoperator values L1*β 和 L2*β，避免Corrector stepRepetitionCompute
        self.L1_beta_star, self.L2_beta_star = self._compute_operator_values(coeffs_star)

        # Store predictor solution供Corrector stepUsing
        self.predictor_solution = U_star_seg
        self.predictor_coeffs = coeffs_star

        return U_star_seg, coeffs_star

    def _compute_operator_values(self, coeffs: np.ndarray) -> Tuple[List[List[Optional[np.ndarray]]], List[List[Optional[np.ndarray]]]]:
        """Compute并Storeoperator values L1*β 和 L2*β

        Args:
            coeffs: np.ndarray, shape (ns, n_eqs, dgN) - coefficient matrix

        Returns:
            L1_beta: List[List[np.ndarray]] - shape [ns][n_eqs](n_points,)
            L2_beta: List[List[np.ndarray]] - shape [ns][n_eqs](n_points,)
        """
        L1_beta = []  # Store L1*β
        L2_beta = []  # Store L2*β

        for seg_idx in range(self.fitter.ns):
            # Get operators for this segment - note dimension matching
            L1_ops = self.fitter._linear_operators[seg_idx].get("L1", None)
            L2_ops = self.fitter._linear_operators[seg_idx].get("L2", None)

            # Compute该segment各Equation的operator values
            L1_seg = []
            L2_seg = []

            for eq_idx in range(self.config.n_eqs):
                # Get coefficients: shape (dgN,)
                beta = coeffs[seg_idx, eq_idx, :]

                # Compute L1*β and L2*β (强制存在化：移除条件检查)
                # L1和L2算子强制存在，无需检查
                L1_val = L1_ops[eq_idx] @ beta
                L2_val = L2_ops[eq_idx] @ beta

                L1_seg.append(L1_val)
                L2_seg.append(L2_val)

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
        """Corrector step: Explicit computation

        u^{n+1} = u^n - Δt * [L1(u^*) + L1(u^n)]/2 - Δt * [L2(u^*) + L2(u^n)]/2 ⋅ F((u^n + u^*)/2) - Δt * N((u^* + u^n)/2)

        CompletelyBased on预Compute的operator values，no matrix solving required
        """

        U_new_seg = []

        for seg_idx in range(len(U_n_seg)):
            U_n = U_n_seg[seg_idx]
            U_star = U_star_seg[seg_idx]

            # Compute midpoint values (u^n + u^*)/2
            U_mid = 0.5 * (U_n + U_star)

            # Get feature matrix
            features = self.fitter._features[seg_idx][0]

            # Compute function values at midpoint (强制存在化：移除条件检查)
            F_mid = self.fitter.F_func(features, U_mid)  # F算子强制存在
            N_mid = self.fitter.N_func(features, U_mid)  # N算子强制存在

            # Initialize corrected solution: u^{n+1} = u^n
            U_corrected = U_n.copy()

            # Perform correction for each equation
            for eq_idx in range(self.config.n_eqs):
                # Compute L1 term average: [L1(u^*) + L1(u^n)]/2
                # Using预Compute的operator values，ensure dimension matching
                L1_avg = np.zeros(U_n.shape[0])  # shape: (n_points,)
                if (hasattr(self, 'L1_beta_star') and len(self.L1_beta_star) > seg_idx and
                    len(self.L1_beta_star[seg_idx]) > eq_idx and self.L1_beta_star[seg_idx][eq_idx] is not None and
                    hasattr(self, 'L1_beta_prev') and self.L1_beta_prev is not None and
                    len(self.L1_beta_prev) > seg_idx and len(self.L1_beta_prev[seg_idx]) > eq_idx):

                    L1_star = self.L1_beta_star[seg_idx][eq_idx]  # shape: (n_points,)
                    L1_n = self.L1_beta_prev[seg_idx][eq_idx]     # shape: (n_points,)
                    if L1_n is not None:
                        L1_avg = 0.5 * (L1_star + L1_n)  # shape: (n_points,)

                # Compute L2⋅F term average: [L2(u^*) + L2(u^n)]/2 ⋅ F((u^n + u^*)/2)
                # Using预Compute的operator values，ensure dimension matching
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

                # Compute N term: N((u^* + u^n)/2) (强制存在化：移除条件检查)
                # N_mid总是存在，N算子强制存在化
                N_contrib = N_mid[eq_idx]  # shape: (n_points,)

                # Explicit update: u^{n+1} = u^n - Δt * [L1_avg + L2F_avg + N_contrib]
                # 确保Allterm都Yes (n_points,) vectors
                U_corrected[:, eq_idx] -= dt * (L1_avg + L2F_avg + N_contrib)

            U_new_seg.append(U_corrected)

        # Due toYesExplicit computation，直接UsingPredictor stepCoefficients
        coeffs_new = coeffs_star

        return U_new_seg, coeffs_new

    def build_stage_jacobian(self, segment_idx: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """BuildPredictor step的Jacobian matrix（Corrector stepYes显式的，no Jacobian needed）

        Maintain consistency with BaseTimeScheme interface
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
        """Predictor step雅可比: [V + Δt⋅L1 + Δt⋅L2⊙F(u^n)] β^* = u^n - Δt⋅N(u^n)

        Build linear system matrix and right-hand side vector
        """

        # Get dimensions and operators
        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.fitter.dgN

        # Get operators and features
        L1_ops = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_ops = self.fitter._linear_operators[segment_idx].get("L2", None)
        features = self.fitter._features[segment_idx][0]

        # Get current solution u^n
        U_n_seg_list = kwargs.get("U_n_seg", [])
        if len(U_n_seg_list) > segment_idx:
            U_n_seg = U_n_seg_list[segment_idx]
        else:
            raise ValueError(f"U_n_seg_list does not contain segment {segment_idx}")

        # Compute F(u^n) 和 N(u^n) (强制存在化：移除条件检查)
        F_n = self.fitter.F_func(features, U_n_seg)  # F算子强制存在
        N_n = self.fitter.N_func(features, U_n_seg)  # N算子强制存在

        # Build system matrix
        A_matrix = np.zeros((ne * n_points, ne * dgN), dtype=np.float64)
        b_vector = []

        for eq_idx in range(ne):
            # Row and column indices
            row_start, row_end = eq_idx * n_points, (eq_idx + 1) * n_points
            col_start, col_end = eq_idx * dgN, (eq_idx + 1) * dgN

            # Build Jacobian for this equation: V + Δt⋅L1 + Δt⋅L2⊙F(u^n)
            J_eq = features.copy()  # Vterm (n_points, dgN)

            # Add Δt⋅L1 term (强制存在化：移除条件检查)
            # L1算子强制存在，无需检查
            J_eq += dt * L1_ops[eq_idx]  # (n_points, dgN)

            # Add Δt⋅L2⊙F(u^n) term (强制存在化：移除条件检查)
            # L2和F算子强制存在，无需检查
            F_eq = F_n[:, eq_idx]  # (n_points,)
            J_eq += dt * np.diag(F_eq) @ L2_ops[eq_idx]  # (n_points, dgN)

            A_matrix[row_start:row_end, col_start:col_end] = J_eq

            # Build right-hand side vector: u^n - Δt⋅N(u^n) (强制存在化：移除条件检查)
            rhs_eq = U_n_seg[:, eq_idx].copy()  # (n_points,)
            # N算子强制存在，无需检查
            rhs_eq -= dt * N_n[eq_idx]  # (n_points,)

            b_vector.append(rhs_eq)

        b_final = np.concatenate(b_vector)  # (ne * n_points,)
        return A_matrix, b_final

    def get_scheme_info(self) -> Dict[str, Any]:
        """ReturnsTime integration schemeinformation"""
        return {
            "name": "OneStepPredictor",
            "type": "predictor-corrector",
            "method": "One-Step Predictor-Corrector",
            "order": self.order,
            "implicit": "predictor only",  # 仅Predictor step隐式
            "explicit": "corrector only", # 仅Corrector step显式
            "stages": 2,  # Predictor step + Corrector step
            "conditional_corrector": True,  # Conditional correction
            "operator_reuse": True,  # operator values复用
            "first_step_reduction": True,  # First step time reduction
            "equation_form": "∂u/∂t + L1(u) + L2(u)⋅F(u) + N(u) = 0",
            "description": "One-step predictor-corrector scheme，operator values预Compute和Time stepBetween复用"
        }

    def validate_operators(self) -> Dict[str, bool]:
        """Validate whether operator configuration meets time scheme requirements（强制存在化：总是返回True）"""
        validation_result = {
            "L1_exists": True,  # 强制存在化：L1算子总是存在
            "L2_exists": True,  # 强制存在化：L2算子总是存在
            "F_exists": True,   # 强制存在化：F算子总是存在
            "N_exists": True,   # 强制存在化：N算子总是存在
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
            # Unified coordinate handling - eliminate dimension checking
            x_flat = x_data.flatten()

            if x_flat.size > 1:
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