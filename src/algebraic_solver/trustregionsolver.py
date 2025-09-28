
import numpy as np
import cupy as cp
from typing import List, Dict, Tuple, Optional
from scipy.sparse import linalg as splinalg

class TrustRegionSolver:
    """Optimize版信赖域Solve器"""
    
    def __init__(self, use_gpu: bool = True, precondition: bool = True):
        """
        Args:
            use_gpu: YesNoUsingGPU加速
            precondition: YesNoUsing预Process
        """
        self.use_gpu = use_gpu and cp.is_available()
        self.precondition = precondition
        self._last_step = None  # 缓存Up一步Result用于热Start
        
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
        initial_guess: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """UsingEnhancement的信赖域法SolveNonlinearSystem
        
        Enhancement:
        1. Using自适应信赖域Radius
        2. 采用预ProcessTechniqueImprovementConvergent性
        3. Implementation热StartStrategy
        4. OptimizeMemory usage
        5. EnhancementLineSearchStrategy
        """
        # InitializeOptimizeParameter
        eta1, eta2 = 0.1, 0.9  # 更宽松的信赖域UpdateThreshold
        radius = 2.0  # 更大的初Beginning信赖域Radius
        radius_max = 20.0
        radius_min = 1e-6
        
        # 智能Initialize
        ns, n_eqs, dgN = shape
        if initial_guess is not None:
            x = initial_guess.reshape(-1, 1)
        elif self._last_step is not None:
            x = self._last_step
        else:
            x = np.zeros((dgN * ns * n_eqs, 1))
            
        # 预AllocateInner存
        if self.use_gpu:
            x_gpu = cp.asarray(x)
            workspace = {"x": x_gpu}
        
        # record最佳Solution
        best_x = x.copy()
        best_residual = float('inf')
        
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
                
            # Build预Process的System
            g = J.T @ f_val
            B = J.T @ J
            
            if self.precondition:
                # Using对CornerLine预Process
                D = np.diag(np.sqrt(np.diag(B) + 1e-12))
                D_inv = np.diag(1.0 / np.diag(D))
                B_scaled = D_inv @ B @ D_inv
                g_scaled = D_inv @ g
            else:
                B_scaled, g_scaled = B, g
                
            # Solve信赖域子Problem
            if self.use_gpu:
                delta = self._solve_trust_region_subproblem_gpu(
                    B_scaled, g_scaled.flatten(), radius, workspace
                )
            else:
                delta = self._solve_trust_region_subproblem(
                    B_scaled, g_scaled.flatten(), radius
                )
                
            # Restore预Process的Step size
            if self.precondition:
                delta = D_inv @ delta
            delta = delta.reshape(-1, 1)
            
            # 自适应Step sizeControl
            step_norm = float(np.linalg.norm(delta))
            if step_norm > radius:
                delta *= (radius / step_norm)
            
            # Compute实际和Prediction减A small amount of
            new_f_val, _ = build_jacobian_fn(x + delta, A, b, equations, variables)
            actual_reduction = f_norm - float(np.linalg.norm(new_f_val))
            predicted_reduction = float(f_norm - np.linalg.norm(f_val + J @ delta))
            
            # ComputeEnhancementRate
            rho = actual_reduction / (predicted_reduction + 1e-10)
            
            # 自适应信赖域Update
            if rho < eta1:
                radius = max(0.25 * radius, radius_min)
            elif rho > eta2 and step_norm >= 0.95 * radius:
                radius = min(2.0 * radius, radius_max)
            elif eta1 <= rho <= eta2:
                radius = max(0.5 * radius, radius_min)
                
            # Step sizeAcceptCriterion
            if rho > 0.05:  # 更宽松的AcceptCondition
                x = x + delta
                
            # Output诊断information
            print(f"Iteration {iter+1}")
            print(f"Trust radius: {radius:.6f}")
            print(f"Step size: {step_norm:.6f}")
            print(f"Improvement ratio: {rho:.6f}")
            print(f"Residual: {np.mean(np.abs(f_val)):.6f}\n")
            
            # ConvergentCheck
            if step_norm < tol * (1 + np.linalg.norm(x)):
                break
                
        # SaveFinally的Result用于热Start
        self._last_step = best_x
        
        return best_x.reshape(ns, n_eqs, dgN)
        
    def _solve_trust_region_subproblem(self, B, g, radius, max_iter=None, tol=1e-8):
        """Enhancement的CPUVersion信赖域子ProblemSolve器"""
        n = len(g)
        max_iter = n if max_iter is None else max_iter
        
        # Attempt柯西point
        eigen_min = splinalg.eigsh(B, k=1, which='SA', return_eigenvectors=False)[0]
        if eigen_min > 0:
            try:
                # Attempt直接Solve
                delta = np.linalg.solve(B, -g)
                if np.linalg.norm(delta) <= radius:
                    return delta
            except np.linalg.LinAlgError:
                pass
                
        # Return退To截断共轭Gradient法
        x = np.zeros_like(g)
        r = g.copy()
        p = -r.copy()
        
        r_norm_sq = r.dot(r)
        p_norm_sq = p.dot(p)
        x_norm_sq = 0.0
        
        for i in range(max_iter):
            Bp = B @ p
            pBp = p.dot(Bp)
            
            # Process负Curvature
            if pBp <= 0:
                # Solve二次Equation
                a = p_norm_sq
                b = 2 * x.dot(p)
                c = x_norm_sq - radius**2
                
                disc = max(b**2 - 4 * a * c, 0)
                tau = (-b + np.sqrt(disc)) / (2 * a)
                return x + tau * p
                
            alpha = r_norm_sq / (pBp + 1e-15)
            x_new = x + alpha * p
            
            # BoundaryCheck
            if np.linalg.norm(x_new) > radius:
                a = p_norm_sq
                b = 2 * x.dot(p)
                c = x_norm_sq - radius**2
                disc = max(b**2 - 4 * a * c, 0)
                tau = (-b + np.sqrt(disc)) / (2 * a)
                return x + tau * p
                
            x = x_new
            x_norm_sq = x.dot(x)
            r_new = r + alpha * Bp
            r_norm_sq_new = r_new.dot(r_new)
            
            # ConvergentCheck
            if np.sqrt(r_norm_sq_new) < tol * np.sqrt(r_norm_sq):
                return x
                
            beta = r_norm_sq_new / r_norm_sq
            r = r_new
            r_norm_sq = r_norm_sq_new
            p = -r + beta * p
            p_norm_sq = r.dot(r) + beta**2 * p_norm_sq + 2 * beta * r.dot(p)
            
        return x
        
    def _solve_trust_region_subproblem_gpu(self, B, g, radius, workspace, max_iter=None, tol=1e-8):
        """Enhancement的GPUVersion信赖域子ProblemSolve器"""
        if 'B' not in workspace:
            workspace['B'] = cp.asarray(B)
            workspace['g'] = cp.asarray(g)
        else:
            # 重用已Allocate的Inner存
            workspace['B'][:] = cp.asarray(B)
            workspace['g'][:] = cp.asarray(g)
            
        B_gpu = workspace['B']
        g_gpu = workspace['g']
        
        n = len(g)
        max_iter = n if max_iter is None else max_iter
        
        # UsingGPUEnter行EigenvalueCompute
        try:
            eigen_min = float(cp.linalg.eigvalsh(B_gpu)[0])
            if eigen_min > 0:
                try:
                    delta = cp.linalg.solve(B_gpu, -g_gpu)
                    if cp.linalg.norm(delta) <= radius:
                        return cp.asnumpy(delta)
                except cp.linalg.LinAlgError:
                    pass
        except cp.linalg.LinAlgError:
            pass
            
        # Initialize
        x = cp.zeros_like(g_gpu)
        r = g_gpu.copy()
        p = -r.copy()
        
        r_norm_sq = float(cp.dot(r, r))
        p_norm_sq = float(cp.dot(p, p))
        x_norm_sq = 0.0
        
        for i in range(max_iter):
            Bp = B_gpu @ p
            pBp = float(cp.dot(p, Bp))
            
            if pBp <= 0:
                a = p_norm_sq
                b = 2 * float(cp.dot(x, p))
                c = x_norm_sq - radius**2
                
                disc = max(b**2 - 4 * a * c, 0)
                tau = (-b + np.sqrt(disc)) / (2 * a)
                return cp.asnumpy(x + tau * p)
                
            alpha = r_norm_sq / (pBp + 1e-15)
            x_new = x + alpha * p
            x_new_norm = float(cp.linalg.norm(x_new))
            
            if x_new_norm > radius:
                a = p_norm_sq
                b = 2 * float(cp.dot(x, p))
                c = x_norm_sq - radius**2
                disc = max(b**2 - 4 * a * c, 0)
                tau = (-b + np.sqrt(disc)) / (2 * a)
                return cp.asnumpy(x + tau * p)
                
            x = x_new
            x_norm_sq = x_new_norm**2
            r_new = r + alpha * Bp
            r_norm_sq_new = float(cp.dot(r_new, r_new))
            
            if cp.sqrt(r_norm_sq_new) < tol * cp.sqrt(r_norm_sq):
                return cp.asnumpy(x)
                
            beta = r_norm_sq_new / r_norm_sq
            r = r_new
            r_norm_sq = r_norm_sq_new
            p = -r + beta * p
            p_norm_sq = float(cp.dot(r, r)) + beta**2 * p_norm_sq + 2 * beta * float(cp.dot(r, p))
            
        return cp.asnumpy(x)