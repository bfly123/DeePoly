"""
FixIterationPDEFitter - Fixed-Point Iteration Fitter

Solves steady-state PDEs using fixed-point iteration:
    L1(U^{n+1}) + L2(U^{n+1})·F(U^{n+1}) + N(U^n) = S
where N(U^n) is fixed from previous iteration.
"""

from typing import Dict, Tuple
import numpy as np

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver


class FixIterationPDEFitter(BaseDeepPolyFitter):
    """Fixed-point iteration fitter for steady-state PDEs

    Core idea: Fix nonlinear term N(U^n), solve linear system for U^{n+1}
    """

    def __init__(self, config, data: Dict = None):
        super().__init__(config, data=data)

        # Initialize solver
        self.solver = LinearSolver(
            verbose=True, use_gpu=True, performance_tracking=True
        )

        # Convergence parameters - read directly from config (already processed by config.py)
        self.max_iterations = getattr(config, "max_iterations", 100)
        self.convergence_tol = getattr(config, "residual_tol", 1e-6)
        self.print_interval = getattr(config, "print_interval", 10)

        # History
        self.residual_history = []

    def _build_segment_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build segment Jacobian for fixed-point iteration

        Steady-state problem: L1(U) + L2(U)·F(U) + N(U) = S
        Fix N(U^n) and F(U^n), solve: [L1 + L2⊙F(U^n)] β^{n+1} = S - N(U^n)

        Args:
            segment_idx: Segment index
            **kwargs: Must contain 'U_current' - current iteration solution

        Returns:
            J: Jacobian matrix (ne*n_points, ne*dgN)
            b: Right-hand side vector (ne*n_points,)
        """
        # Get current solution from kwargs
        U_current = kwargs.get("U_current")
        if U_current is None:
            raise ValueError(
                "U_current must be provided in kwargs for fixed-point iteration"
            )

        # ============= DIAGNOSTIC 0: Input U_current =============
        if segment_idx == 0:  # Only print for first segment
            print(f"\n{'='*70}")
            print(f"🔍 DEBUG: _build_segment_jacobian (Segment {segment_idx})")
            print(f"{'='*70}")
            print(f"U_current shape: {U_current.shape}")
            print(f"U_current range: [{np.min(U_current):.6f}, {np.max(U_current):.6f}]")
            print(f"U_current mean: {np.mean(U_current):.6f}, std: {np.std(U_current):.6f}")

        # Convert to segment format
        U_seg_list = self.global_to_segments(U_current)
        U_seg = U_seg_list[segment_idx]

        # ============= DIAGNOSTIC 1: Current solution values =============
        if segment_idx == 0:  # Only print for first segment to reduce output
            print(f"\nSegment info:")
            n_points = len(self.data["x_segments_norm"][segment_idx])
            ne = self.config.n_eqs
            dgN = self.dgN
            print(f"  n_points: {n_points}, n_eqs: {ne}, dgN: {dgN}")
            print(f"  U_seg shape: {U_seg.shape}")
            print(f"  U_seg range: [{np.min(U_seg):.6f}, {np.max(U_seg):.6f}]")

        # Get dimensions
        n_points = len(self.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.dgN

        # Get precompiled operators and features
        L1_operators = self._linear_operators[segment_idx].get("L1", None)
        L2_operators = self._linear_operators[segment_idx].get("L2", None)
        features_list = self._features[segment_idx]
        features = features_list[0]  # 0-th order derivative

        # Compute fixed nonlinear terms F(U^n) and N(U^n)
        F_n = self.F_func(features, U_seg)  # Shape: (n_points, ne)
        N_n_list = self.N_func(features, U_seg)  # List of ne arrays

        # ============= DIAGNOSTIC 2: Nonlinear terms F and N =============
        if False and segment_idx == 0:
            print(f"\n[DIAGNOSTIC 2 - Segment {segment_idx}] Nonlinear terms:")
            print(f"  F_n shape: {F_n.shape}")
            print(f"  F_n range: [{np.min(F_n):.6e}, {np.max(F_n):.6e}]")
            for eq_idx, n_val in enumerate(N_n_list):
                n_arr = np.array(n_val).flatten()
                print(f"  N_n[{eq_idx}] range: [{np.min(n_arr):.6e}, {np.max(n_arr):.6e}]")
                print(f"  N_n[{eq_idx}] mean: {np.mean(n_arr):.6e}")

        # Convert N from list to array format (n_points, ne)
        N_n = np.zeros((n_points, ne))
        for eq_idx, n_val in enumerate(N_n_list):
            N_n[:, eq_idx] = np.array(n_val).flatten()[:n_points]

        # Get source term
        S = self._get_source_term(segment_idx)

        # ============= DIAGNOSTIC 3: Source term =============
        if False and segment_idx == 0:
            print(f"\n[DIAGNOSTIC 3 - Segment {segment_idx}] Source term:")
            print(f"  S shape: {S.shape}")
            print(f"  S range: [{np.min(S):.6e}, {np.max(S):.6e}]")
            print(f"  S norm: {np.linalg.norm(S):.6e}")

        # Build system matrix and RHS
        L_final = np.zeros((ne * n_points, ne * dgN))
        b_final = []

        # Build Jacobian for each equation
        for eq_idx in range(ne):
            L1 = L1_operators[eq_idx]
            L2 = L2_operators[eq_idx]

            # Row index range
            row_start = eq_idx * n_points
            row_end = (eq_idx + 1) * n_points

            # Column index range
            col_start = eq_idx * dgN
            col_end = (eq_idx + 1) * dgN

            # Build Jacobian for this equation: L1 + L2⊙F(U^n)
            # Note: No V (mass matrix) term for steady-state problem
            # Start with L1
            J_eq = L1.copy()

            # Add L2⊙F term: diag(F) @ L2
            F_eq = F_n[:, eq_idx]
            L2F_term = np.diag(F_eq) @ L2
            J_eq += L2F_term

            # ============= DIAGNOSTIC 4: Jacobian components for each equation =============
            if False and segment_idx == 0:
                print(f"\n[DIAGNOSTIC 4 - Segment {segment_idx}, Eq {eq_idx}] Jacobian components:")
                print(f"  L1 shape: {L1.shape}, nnz ratio: {np.count_nonzero(L1)/L1.size:.4f}")
                print(f"  L1 norm: {np.linalg.norm(L1, 'fro'):.6e}")
                print(f"  L2 shape: {L2.shape}, nnz ratio: {np.count_nonzero(L2)/L2.size:.4f}")
                print(f"  L2 norm: {np.linalg.norm(L2, 'fro'):.6e}")
                print(f"  F_eq (current iter) range: [{np.min(F_eq):.6e}, {np.max(F_eq):.6e}]")
                print(f"  L2F term norm: {np.linalg.norm(L2F_term, 'fro'):.6e}")
                print(f"  J_eq total norm: {np.linalg.norm(J_eq, 'fro'):.6e}")

            # Place in final matrix
            L_final[row_start:row_end, col_start:col_end] = J_eq

            # Build RHS for this equation: S - N(U^n)
            rhs_eq = S[:, eq_idx] - N_n[:, eq_idx]

            if False and segment_idx == 0:
                print(f"  RHS: S - N norm: {np.linalg.norm(rhs_eq):.6e}")
                print(f"  RHS range: [{np.min(rhs_eq):.6e}, {np.max(rhs_eq):.6e}]")

            b_final.append(rhs_eq)

        # Flatten RHS vector
        b_vector = np.concatenate(b_final)

        # ============= DIAGNOSTIC 5: Final Jacobian summary =============
        if False and segment_idx == 0:
            print(f"\n[DIAGNOSTIC 5 - Segment {segment_idx}] Final Jacobian:")
            print(f"  L_final shape: {L_final.shape}")
            print(f"  L_final norm: {np.linalg.norm(L_final, 'fro'):.6e}")
            print(f"  L_final nnz ratio: {np.count_nonzero(L_final)/L_final.size:.4f}")
            print(f"  b_vector norm: {np.linalg.norm(b_vector):.6e}")
            print(f"  b_vector range: [{np.min(b_vector):.6e}, {np.max(b_vector):.6e}]")

        return L_final, b_vector
