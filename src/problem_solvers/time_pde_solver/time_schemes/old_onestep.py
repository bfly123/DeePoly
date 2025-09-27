"""IMEX Predictor-Corrector time integration scheme based on Allen-Cahn standalone solver"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from .base_time_scheme import BaseTimeScheme


class ImexPredictorCorrector(BaseTimeScheme):
    """IMEX Predictor-Corrector time integration scheme
    
    Based on the time stepping approach from HBNet_1D_PDE_time_NetCut_2nd_secondversion_AC.py:
    
    Time evolution equation:
    ∂U/∂t = L1(U) + L2(U)⊙F(U) + N(U)
   
    IMEX Predictor-Corrector scheme:
    
    Predictor (implicit for stiff terms):
    [V + dt*L1] β* = u^n - dt*(L2(u^n)⊙F(u^n))
    
    Corrector (explicit update):
    - For it = 1: u^{n+1} = u^n - dt*(L1(u*) + L2(u^n)⊙F(u^n))
    - For it > 1: u^{n+1} = u^n - dt*(L1_avg + L2(u*)⊙F(u*))
      where L1_avg uses averaged derivatives from current and previous steps
    """

    def __init__(self, config):
        super().__init__(config)
        
        self.order = 1  # First-order method
        self.method_name = "IMEX Predictor-Corrector"
        
        # Storage for derivative averaging
        self.du_x_old = None
        self.du_xx_old = None
        self.time_step_counter = 0
        
        # Storage for current segment-level solution values
        self.U_seg_current = None
        self.U_seg_prev = None

    def time_step(
        self, U_n: np.ndarray, U_seg: List[np.ndarray], dt: float, coeffs_n: np.ndarray = None, current_time: float = 0.0, step: int = 0
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute IMEX Predictor-Corrector time step"""
        
        # Store current time for plotting
        self._current_time = current_time
        self.time_step_counter = step
        
        # Initialize or update current segment-level solution values
        self.U_seg_n = U_seg.copy()
        
        # Step 1: Predictor step (implicit for stiff L1 terms)
        args = {
            "dt": dt,
            "stage": "predictor",
            "step": step,
            "U_n_seg": self.U_seg_n,
            "operation": "imex_predictor"
        }
        U_seg_predictor, coeffs_predictor = self._solve_imex_predictor_stage(**args)
        
        # Step 2: Corrector step (explicit update)
        # Extract derivatives from predictor for corrector step
        du_x_current, du_xx_current = self._extract_derivatives_from_coeffs(coeffs_predictor)
        
        # Apply corrector update
        U_seg_new = self._apply_corrector_update(
            U_seg_n=self.U_seg_n,
            U_seg_predictor=U_seg_predictor,
            du_x_current=du_x_current,
            du_xx_current=du_xx_current,
            dt=dt,
            step=step
        )
        
        # Store derivatives for next time step
        self.du_x_old = copy.deepcopy(du_x_current)
        self.du_xx_old = copy.deepcopy(du_xx_current)
        
        # Convert back to global array
        U_new = self.fitter.segments_to_global(U_seg_new)
        
        return (
            U_new,
            U_seg_new,
            coeffs_predictor,  # Return predictor coefficients
        )

    def _solve_imex_predictor_stage(self, **kwargs) -> Tuple[List[np.ndarray], np.ndarray]:
        """Solve IMEX predictor stage (implicit for stiff terms)"""
        
        # Prepare stage data
        stage_data = {
            "U_n_seg": kwargs.get("U_n_seg"),
            "dt": kwargs.get("dt"),
            "stage": kwargs.get("stage"),
            "operation": kwargs.get("operation", "imex_predictor"),
            "step": kwargs.get("step"),
        }
        
        # Use base_fitter's fit method to solve for coefficients
        coeffs_predictor = self.fitter.fit(**stage_data)
        
        # Convert coefficients to solution values
        U_predictor_global, U_predictor_segments = self.fitter.construct(
            self.fitter.data, self.fitter._current_model, coeffs_predictor
        )
        
        return U_predictor_segments, coeffs_predictor

    def _extract_derivatives_from_coeffs(self, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract first and second derivatives from coefficients"""
        
        du_x_list = []
        du_xx_list = []
        
        for segment_idx in range(self.fitter.ns):
            # Get features for different derivative orders
            features_list = self.fitter._features[segment_idx]
            
            # Extract coefficients for this segment
            beta = coeffs[segment_idx, 0, :]  # Assuming single equation (Allen-Cahn)
            
            # Compute derivatives using features
            if len(features_list) >= 3:
                du_x_seg = features_list[1] @ beta  # 1st derivative
                du_xx_seg = features_list[2] @ beta  # 2nd derivative
                
                du_x_list.append(du_x_seg.flatten())
                du_xx_list.append(du_xx_seg.flatten())
        
        # Concatenate all segments
        du_x_global = np.concatenate(du_x_list) if du_x_list else np.array([])
        du_xx_global = np.concatenate(du_xx_list) if du_xx_list else np.array([])
        
        return du_x_global, du_xx_global

    def _apply_corrector_update(
        self, 
        U_seg_n: List[np.ndarray],
        U_seg_predictor: List[np.ndarray], 
        du_xx_current: np.ndarray,
        dt: float, 
        step: int
    ) -> List[np.ndarray]:
        """Apply corrector update step"""
        
        U_seg_new = []
        
        start_idx = 0
        for segment_idx in range(self.fitter.ns):
            U_n_seg = U_seg_n[segment_idx]
            U_pred_seg = U_seg_predictor[segment_idx]
            n_points = U_n_seg.shape[0]
            
            # Extract segment derivatives
            end_idx = start_idx + n_points
            du_xx_seg = du_xx_current[start_idx:end_idx] if len(du_xx_current) > end_idx else np.zeros(n_points)
            
            if step > 1:
                # Use averaged derivatives for corrector (it > 1)
                du_xx_old_seg = self.du_xx_old[start_idx:end_idx] if self.du_xx_old is not None and len(self.du_xx_old) > end_idx else np.zeros(n_points)
                du_xx_avg = (du_xx_seg + du_xx_old_seg) / 2

                # Average predictor and previous solution for nonlinear term
                U_avg_seg = (U_pred_seg + U_n_seg) / 2

                # Corrector update: u^{n+1} = u^n - dt*(L1_avg + L2(u*)⊙F(u*))
                # L1_avg = -0.0001 * du_xx_avg (diffusion)
                L1_term = -0.0001 * du_xx_avg

                # 调试：对比factory生成的L1算子
                if step <= 2 and segment_idx == 0:  # 只在第一个segment打印
                    print(f"\n--- Step {step}, Segment {segment_idx} ---")
                    # 获取factory生成的L1算子
                    L1_ops = self.fitter._linear_operators[segment_idx].get("L1", None)
                    if L1_ops is not None and len(L1_ops) > 0:
                        # 从predictor coeffs计算L1值
                        coeffs_segment = self.du_x_old  # 这里实际需要coeffs，暂用占位
                        print(f"Factory L1 operator shape: {L1_ops[0].shape}")
                        # 计算一个样本点的值
                        print(f"硬编码 L1_term[0:3]: {L1_term[0:3]}")
                        print(f"硬编码公式: -0.0001 * du_xx_avg")
                        print(f"du_xx_avg[0:3]: {du_xx_avg[0:3]}")
                
                # L2⊙F = u* * (5 - 5*(u*)²) (reaction)
                U_avg_flat = U_avg_seg.flatten()
                L2F_term = U_avg_flat * (5 - 5 * U_avg_flat**2)

                # 调试：对比N算子和硬编码的L2F_term
                if step <= 2 and segment_idx == 0:
                    # 计算factory的N算子值
                    features = self.fitter._features[segment_idx][0]
                    if self.fitter.has_operator("N"):
                        N_vals = self.fitter.N_func(features, U_avg_seg)
                        if isinstance(N_vals, np.ndarray):
                            N_factory = N_vals[:, 0] if N_vals.ndim > 1 else N_vals
                            print(f"\n对比非线性项：")
                            print(f"硬编码 L2F_term[0:3]: {L2F_term[0:3]}")
                            print(f"硬编码公式: u*(5-5*u²) = 5*u - 5*u³")
                            print(f"Factory N算子[0:3]: {N_factory[0:3]}")
                            print(f"Factory定义: N = 5*u³ - 5*u (config.json)")
                            print(f"差异: {(L2F_term - N_factory)[0:3]}")
                            print(f"U_avg_flat[0:3]: {U_avg_flat[0:3]}")

                # Apply update
                U_new_seg = U_n_seg.copy()
                U_new_seg[:, 0] = U_n_seg[:, 0] - dt * (L1_term + L2F_term)
                
            else:
                # First step: u^{n+1} = u^n - dt*(L1(u*) + L2(u^n)⊙F(u^n))
                # L1 term from predictor
                L1_term = -0.0001 * du_xx_seg

                # L2⊙F term using u^n
                U_n_flat = U_n_seg.flatten()
                L2F_term = U_n_flat * (5 - 5 * U_n_flat**2)

                # 调试：在第一步也对比
                if step == 1 and segment_idx == 0:
                    print(f"\n--- Step {step}, Segment {segment_idx} (First Step) ---")
                    features = self.fitter._features[segment_idx][0]
                    if self.fitter.has_operator("N"):
                        N_vals = self.fitter.N_func(features, U_n_seg)
                        if isinstance(N_vals, np.ndarray):
                            N_factory = N_vals[:, 0] if N_vals.ndim > 1 else N_vals
                            print(f"\n对比非线性项：")
                            print(f"硬编码 L2F_term = u*(5-5*u²) = 5*u - 5*u³")
                            print(f"  值[0:3]: {L2F_term[0:3]}")
                            print(f"Factory N = 5*u³ - 5*u (来自config.json)")
                            print(f"  值[0:3]: {N_factory[0:3]}")
                            print(f"差异[0:3]: {(L2F_term - N_factory)[0:3]}")
                            print(f"差异是否为-2*N: {np.allclose(L2F_term, -N_factory)}")
                            print(f"U_n_flat[0:3]: {U_n_flat[0:3]}")

                # Apply update
                U_new_seg = U_n_seg.copy()
                U_new_seg[:, 0] = U_n_seg[:, 0] - dt * (L1_term + L2F_term)
            
            U_seg_new.append(U_new_seg)
            start_idx = end_idx
        
        return U_seg_new

    def build_stage_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build IMEX Predictor-Corrector stage Jacobian matrix
        
        For predictor stage (implicit):
        [V + dt*L1] β* = u^n - dt*(L2(u^n)⊙F(u^n))
        
        Where:
        - V: features matrix
        - L1: [-0.0001 * ∂²/∂x²] (diffusion operator, treated implicitly)
        - L2⊙F: [u^n * (5 - 5*(u^n)²)] (reaction term, treated explicitly in RHS)
        """
        
        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.fitter.dgN
        
        stage = kwargs.get("stage", "predictor")
        dt = kwargs.get("dt")
        step = kwargs.get("step", 1)
        
        # Get precompiled operators and features
        L1_operators = self.fitter._linear_operators[segment_idx].get("L1", None)
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)
        features_list = self.fitter._features[segment_idx]
        features = features_list[0]  # 0th order derivative (features matrix V)
        
        U_n_seg_list = kwargs.get("U_n_seg", [])
        U_n_seg = U_n_seg_list[segment_idx]
        
        # System matrix and RHS
        L_final = np.zeros((ne * n_points, ne * dgN))
        b_final = []
        
        if stage == "predictor":
            # Predictor: [V + dt*L1] β* = RHS
            for eq_idx in range(ne):
                # Row and column indices
                row_start = eq_idx * n_points
                row_end = (eq_idx + 1) * n_points
                col_start = eq_idx * dgN
                col_end = (eq_idx + 1) * dgN
                
                # Start with features matrix V
                J_eq = features.copy()
                
                # Add implicit L1 term: +dt*L1 (diffusion)
                if L1_operators is not None:
                    L1 = L1_operators[eq_idx]
                    J_eq += dt * L1
                
                # Set coefficient matrix
                L_final[row_start:row_end, col_start:col_end] = J_eq
                
                # Build RHS: u^n - dt*(L2(u^n)⊙F(u^n))
                rhs_seg = U_n_seg[:, eq_idx].copy()
                
                # Subtract explicit reaction term
                if L2_operators is not None and self.fitter.has_operator("F"):
                    F_vals = self.fitter.F_func(features, U_n_seg)
                    F_eq = F_vals[:, eq_idx]
                    
                    # L2(u^n) @ some_beta * F(u^n) 
                    # For this explicit treatment, we approximate L2(u^n) ≈ u^n
                    U_n_eq = U_n_seg[:, eq_idx]
                    L2F_term = U_n_eq * F_eq
                    
                    rhs_seg -= dt * L2F_term
                
                b_final.append(rhs_seg)
        
        # Flatten right-hand side vector
        b_vector = np.concatenate(b_final)
        
        return L_final, b_vector

    def get_scheme_info(self) -> Dict[str, Any]:
        """Get IMEX Predictor-Corrector time scheme information"""
        return {
            "method": self.method_name,
            "stages": "Predictor (implicit) + Corrector (explicit)",
            "order": self.order,
            "stability": "IMEX (A-stable for stiff terms, explicit for non-stiff)",
            "operators": {
                "L1": "Stiff diffusion term: -0.0001 * ∂²u/∂x² (implicit)",
                "L2": "Linear multiplicative term: u",
                "F": "Nonlinear source: 5 - 5*u²",
                "N": "No separate explicit nonlinear terms",
            },
            "equation_form": "∂u/∂t = L1(u) + L2(u)⊙F(u)",
            "specific_form": "∂u/∂t = -0.0001*∂²u/∂x² + u*(5-5*u²)",
            "predictor": "[V + dt*L1] β* = u^n - dt*L2(u^n)⊙F(u^n)",
            "corrector": "u^{n+1} = u^n - dt*(L1_avg + L2(u*)⊙F(u*))",
        }

    def validate_operators(self) -> Dict[str, bool]:
        """Validate operator configuration for IMEX Predictor-Corrector scheme"""
        validation = {
            "L1_exists": self.fitter.has_operator("L1"),
            "L2_exists": self.fitter.has_operator("L2"),
            "F_exists": self.fitter.has_operator("F"),
            "N_exists": self.fitter.has_operator("N"),  # Should be False for this scheme
            "precompiled": self.fitter._precompiled,
        }
        
        # Check IMEX Predictor-Corrector requirements
        validation["imex_pc_ready"] = (
            validation["precompiled"] 
            and validation["L1_exists"]  # Need stiff L1 for implicit treatment
            and validation["L2_exists"]  # Need L2 for explicit reaction
            and validation["F_exists"]   # Need F for nonlinear source
        )
        
        return validation

    def estimate_stable_dt(
        self, U_current: np.ndarray, safety_factor: float = 0.8
    ) -> float:
        """Estimate stable time step for IMEX Predictor-Corrector scheme"""
        
        if not self.fitter._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")
        
        max_eigenvalue = 0.0
        
        # For IMEX scheme, stability is mainly limited by explicit terms
        # since stiff diffusion is treated implicitly
        
        # Estimate reaction term stability: L2⊙F = u*(5-5*u²)
        # Jacobian: d/du[u*(5-5*u²)] = 5 - 15*u²
        # Maximum reaction rate occurs at u = 0: df/du = 5
        max_reaction_rate = 5.0
        max_eigenvalue = max(max_eigenvalue, max_reaction_rate)
        
        # For safety, also consider a spatial constraint
        if hasattr(self.fitter, 'data') and 'x_segments_norm' in self.fitter.data:
            min_dx = np.inf
            for segment_idx in range(self.fitter.ns):
                x_seg = self.fitter.data['x_segments_norm'][segment_idx]
                if len(x_seg) > 1:
                    dx_seg = np.min(np.diff(x_seg.flatten()))
                    min_dx = min(min_dx, dx_seg)
            
            if min_dx < np.inf:
                # Conservative spatial constraint
                spatial_constraint = 1.0 / min_dx
                max_eigenvalue = max(max_eigenvalue, spatial_constraint)
        
        if max_eigenvalue <= 0:
            return 0.01  # Default conservative value
        
        # IMEX stability constraint (more relaxed than explicit)
        dt_stable = safety_factor / max_eigenvalue
        
        return min(dt_stable, 0.5)  # Cap at reasonable maximum

    def get_butcher_tableau(self) -> Dict[str, Any]:
        """Get IMEX Predictor-Corrector scheme parameters"""
        return {
            "method_type": "IMEX Predictor-Corrector",
            "predictor": "Implicit for stiff L1 terms",
            "corrector": "Explicit update with derivative averaging",
            "order": self.order,
            "implicit_terms": "L1 (diffusion)",
            "explicit_terms": "L2⊙F (reaction)",
            "note": "Based on HBNet_1D_PDE_time_NetCut_2nd_secondversion_AC.py approach",
        }

    def print_scheme_summary(self):
        """Print IMEX Predictor-Corrector time integration scheme summary"""
        validation = self.validate_operators()
        info = self.get_scheme_info()
        
        print(f"=== {info['method']} Time Integration Summary ===")
        print(f"Method: {info['method']}")
        print(f"Type: {info['stages']}, Order: {info['order']}")
        print(f"Stability: {info['stability']}")
        print(f"Equation form: {info['equation_form']}")
        print(f"Allen-Cahn form: {info['specific_form']}")
        
        print("\nScheme Details:")
        print(f"Predictor: {info['predictor']}")
        print(f"Corrector: {info['corrector']}")
        
        print("\nOperator Status:")
        for op_name, exists in validation.items():
            if op_name.endswith('_exists'):
                op_display = op_name.replace('_exists', '').upper()
                status = '✓' if exists else '✗'
                if op_name == 'N_exists' and not exists:
                    status += ' (expected for this scheme)'
                print(f"  {op_display}: {status}")
        
        ready_key = 'imex_pc_ready'
        if ready_key in validation:
            print(f"\nReady for time integration: {'✓' if validation[ready_key] else '✗'}")
        print("=" * 70)