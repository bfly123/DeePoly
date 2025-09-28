import numpy as np
import cupy as cp
from typing import List, Dict, Tuple, Optional
from scipy.sparse import linalg as splinalg


class GaussNewtonSolver:
    """Optimize的Gauss-NewtonSolve器，UsingLineSearchStrategy"""

    def __init__(self, use_gpu: bool = True, use_linesearch: bool = True):
        """
        Args:
            use_gpu: YesNoUsingGPU加速
            use_linesearch: YesNoUsingLineSearch
        """
        self.use_gpu = use_gpu and cp.is_available()
        self.use_linesearch = use_linesearch
        self._last_solution = None

    def solve(
        self,
        A: List,
        b: List,
        equations: Dict,
        variables: Dict,
        build_jacobian_fn,
        shape: Tuple[int, int, int],
        max_iter: int = 50,
        tol: float = 1e-8,
        initial_guess: Optional[np.ndarray] = None,
        damping: float = 1e-4,  # Levenberg-MarquardtDamping因子
    ) -> np.ndarray:
        """UsingOptimize的Gauss-Newton法SolveNonlinearSystem

        特point:
        1. UsingLevenberg-MarquardtDampingImprovementStability
        2. 采用自适应LineSearch
        3. Implementation预Condition共轭GradientSolveLinear子Problem
        4. UsingQRDecomposeProcess病态Problem
        """
        # 智能Initialize
        ns, n_eqs, dgN = shape
        if initial_guess is not None:
            x = initial_guess.reshape(-1, 1)
        elif self._last_solution is not None:
            x = self._last_solution
        else:
            x = np.zeros((dgN * ns * n_eqs, 1))

        # 预AllocateGPUInner存
        if self.use_gpu:
            workspace = {"x": cp.asarray(x), "buffer": cp.empty_like(x)}

        # record最佳Solution
        best_x = x.copy()
        best_residual = float("inf")

        # WolfeLineSearchParameter
        c1 = 1e-4  # ArmijoConditionParameter
        c2 = 0.9  # CurvatureConditionParameter
        alpha_init = 1.0

        for iter in range(max_iter):
            # ComputeResidual和Jacobian matrix
            f_val, J = build_jacobian_fn(x, A, b, equations, variables)
            f_norm = float(np.linalg.norm(f_val))

            # Update最佳Solution
            if f_norm < best_residual:
                best_residual = f_norm
                best_x = x.copy()

            # ConvergentCheck
            if f_norm < tol:
                break

            # BuildGauss-NewtonSystem
            g = J.T @ f_val

            # UsingQRDecomposeSolve正规Equation
            # 这比直接Solve J^T J 更Stable
            try:
                Q, R = np.linalg.qr(J)
                delta = -np.linalg.solve(R.T @ R + damping * np.eye(R.shape[1]), g)
            except np.linalg.LinAlgError:
                # IfQRDecomposeFail，Return退To预Condition共轭Gradient
                B = J.T @ J + damping * np.eye(J.shape[1])
                delta = self._solve_pcg(B, -g)

            delta = delta.reshape(-1, 1)

            if self.use_linesearch:
                # UsingWolfeLineSearch
                alpha = self._wolfe_linesearch(
                    x,
                    delta,
                    f_val,
                    J,
                    A,
                    b,
                    equations,
                    variables,
                    build_jacobian_fn,
                    c1,
                    c2,
                    alpha_init,
                )
                step = alpha * delta
            else:
                step = delta

            # UpdateSolution
            x_new = x + step

            # Compute实际Enhancement
            new_f_val, _ = build_jacobian_fn(x_new, A, b, equations, variables)
            new_f_norm = float(np.linalg.norm(new_f_val))
            mean_f_norm = np.mean(np.abs(new_f_val))

            # 自适应DampingUpdate
            if new_f_norm < f_norm:
                damping = max(damping * 0.1, 1e-7)  # 减小Damping
                x = x_new
            else:
                damping = min(damping * 10, 1e2)  # IncreaseDamping

            # Output诊断information
            print(f"Iteration {iter+1}")
            print(f"Step size: {float(np.linalg.norm(step)):.6f}")
            print(f"Residual: {new_f_norm:.6f}")
            print(f"Mean Residual: {mean_f_norm:.6f}")
            print(f"Damping: {damping:.6e}\n")

            # ConvergentCheck
            if np.linalg.norm(step) < tol * (1 + np.linalg.norm(x)):
                break

        # SaveResult用于热Start
        self._last_solution = best_x

        return best_x.reshape(ns, n_eqs, dgN)

    def _solve_pcg(self, A, b, tol=1e-10, max_iter=None):
        """预Condition共轭Gradient法SolveLinearSystem"""
        n = len(b)
        max_iter = n if max_iter is None else max_iter

        # 简单的对CornerLine预Condition子
        M = np.diag(np.diag(A))

        x = np.zeros_like(b)
        r = b - A @ x
        z = np.linalg.solve(M, r)
        p = z.copy()

        r_norm_new = r.T @ z

        for i in range(max_iter):
            Ap = A @ p
            alpha = r_norm_new / (p.T @ Ap)

            x += alpha * p
            r -= alpha * Ap

            z = np.linalg.solve(M, r)
            r_norm_old = r_norm_new
            r_norm_new = r.T @ z

            if np.sqrt(abs(r_norm_new)) < tol:
                break

            beta = r_norm_new / r_norm_old
            p = z + beta * p

        return x

    def _wolfe_linesearch(
        self,
        x,
        p,
        f_val,
        J,
        A,
        b,
        equations,
        variables,
        build_jacobian_fn,
        c1,
        c2,
        alpha_init,
        max_iter=10,
    ):
        """WolfeLineSearch以确定最优Step size"""
        phi_0 = float(np.linalg.norm(f_val))
        dphi_0 = float(2 * f_val.T @ (J @ p))

        alpha = alpha_init
        alpha_prev = 0
        phi_prev = phi_0

        for i in range(max_iter):
            # Compute新point的函Numerical
            x_new = x + alpha * p
            f_new, J_new = build_jacobian_fn(x_new, A, b, equations, variables)
            phi = float(np.linalg.norm(f_new))

            # ArmijoConditionCheck
            if phi > phi_0 + c1 * alpha * dphi_0:
                return self._zoom(
                    alpha_prev,
                    alpha,
                    phi_prev,
                    phi,
                    phi_0,
                    dphi_0,
                    x,
                    p,
                    A,
                    b,
                    equations,
                    variables,
                    build_jacobian_fn,
                    c1,
                    c2,
                )

            # Compute新Derivatives
            dphi = float(2 * f_new.T @ (J_new @ p))

            # CurvatureConditionCheck
            if abs(dphi) <= -c2 * dphi_0:
                return alpha

            if dphi >= 0:
                return self._zoom(
                    alpha,
                    alpha_prev,
                    phi,
                    phi_prev,
                    phi_0,
                    dphi_0,
                    x,
                    p,
                    A,
                    b,
                    equations,
                    variables,
                    build_jacobian_fn,
                    c1,
                    c2,
                )

            alpha_prev = alpha
            phi_prev = phi
            alpha *= 2.0

        return alpha

    def _zoom(
        self,
        alpha_lo,
        alpha_hi,
        phi_lo,
        phi_hi,
        phi_0,
        dphi_0,
        x,
        p,
        A,
        b,
        equations,
        variables,
        build_jacobian_fn,
        c1,
        c2,
    ):
        """ZoomPhase的LineSearch"""
        for i in range(10):
            # 二分法选择Newalpha
            alpha = (alpha_lo + alpha_hi) / 2.0

            # Compute新point的函Numerical
            x_new = x + alpha * p
            f_new, J_new = build_jacobian_fn(x_new, A, b, equations, variables)
            phi = float(np.linalg.norm(f_new))

            if phi > phi_0 + c1 * alpha * dphi_0 or phi >= phi_lo:
                alpha_hi = alpha
            else:
                dphi = float(2 * f_new.T @ (J_new @ p))

                if abs(dphi) <= -c2 * dphi_0:
                    return alpha

                if dphi * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo

                alpha_lo = alpha

        return alpha
