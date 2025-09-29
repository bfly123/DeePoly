import numpy as np
import cupy as cp
from typing import List, Dict, Tuple, Optional
from scipy.sparse import linalg as splinalg


class GaussNewtonSolver:
    """优化的高斯-牛顿求解器，使用线搜索策略"""

    def __init__(self, use_gpu: bool = True, use_linesearch: bool = True):
        """
        Args:
            use_gpu: 是否使用GPU加速
            use_linesearch: 是否使用线搜索
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
        damping: float = 1e-4,  # Levenberg-Marquardt阻尼因子
    ) -> np.ndarray:
        """使用优化的高斯-牛顿法求解非线性系统

        特点:
        1. 使用Levenberg-Marquardt阻尼提高稳定性
        2. 采用自适应线搜索
        3. 实现预条件共轭梯度求解线性子问题
        4. 使用QR分解处理病态问题
        """
        # 智能初始化
        ns, n_eqs, dgN = shape
        if initial_guess is not None:
            x = initial_guess.reshape(-1, 1)
        elif self._last_solution is not None:
            x = self._last_solution
        else:
            x = np.zeros((dgN * ns * n_eqs, 1))

        # 预分配GPU内存
        if self.use_gpu:
            workspace = {"x": cp.asarray(x), "buffer": cp.empty_like(x)}

        # 记录最佳解
        best_x = x.copy()
        best_residual = float("inf")

        # Wolfe线搜索参数
        c1 = 1e-4  # Armijo条件参数
        c2 = 0.9  # 曲率条件参数
        alpha_init = 1.0

        for iter in range(max_iter):
            # 计算残差和雅可比矩阵
            f_val, J = build_jacobian_fn(x, A, b, equations, variables)
            f_norm = float(np.linalg.norm(f_val))

            # 更新最佳解
            if f_norm < best_residual:
                best_residual = f_norm
                best_x = x.copy()

            # 收敛检查
            if f_norm < tol:
                break

            # 构建高斯-牛顿系统
            g = J.T @ f_val

            # 使用QR分解求解正规方程
            # 这比直接求解 J^T J 更稳定
            try:
                Q, R = np.linalg.qr(J)
                delta = -np.linalg.solve(R.T @ R + damping * np.eye(R.shape[1]), g)
            except np.linalg.LinAlgError:
                # 如果QR分解失败，回退到预条件共轭梯度
                B = J.T @ J + damping * np.eye(J.shape[1])
                delta = self._solve_pcg(B, -g)

            delta = delta.reshape(-1, 1)

            if self.use_linesearch:
                # 使用Wolfe线搜索
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

            # 更新解
            x_new = x + step

            # 计算实际改进
            new_f_val, _ = build_jacobian_fn(x_new, A, b, equations, variables)
            new_f_norm = float(np.linalg.norm(new_f_val))
            mean_f_norm = np.mean(np.abs(new_f_val))

            # 自适应阻尼更新
            if new_f_norm < f_norm:
                damping = max(damping * 0.1, 1e-7)  # 减小阻尼
                x = x_new
            else:
                damping = min(damping * 10, 1e2)  # 增大阻尼

            # 输出诊断信息
            print(f"Iteration {iter+1}")
            print(f"Step size: {float(np.linalg.norm(step)):.6f}")
            print(f"Residual: {new_f_norm:.6f}")
            print(f"Mean Residual: {mean_f_norm:.6f}")
            print(f"Damping: {damping:.6e}\n")

            # 收敛检查
            if np.linalg.norm(step) < tol * (1 + np.linalg.norm(x)):
                break

        # 保存结果用于热启动
        self._last_solution = best_x

        return best_x.reshape(ns, n_eqs, dgN)

    def _solve_pcg(self, A, b, tol=1e-10, max_iter=None):
        """预条件共轭梯度法求解线性系统"""
        n = len(b)
        max_iter = n if max_iter is None else max_iter

        # 简单的对角线预条件子
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
        """Wolfe线搜索以确定最优步长"""
        phi_0 = float(np.linalg.norm(f_val))
        dphi_0 = float(2 * f_val.T @ (J @ p))

        alpha = alpha_init
        alpha_prev = 0
        phi_prev = phi_0

        for i in range(max_iter):
            # 计算新点的函数值
            x_new = x + alpha * p
            f_new, J_new = build_jacobian_fn(x_new, A, b, equations, variables)
            phi = float(np.linalg.norm(f_new))

            # Armijo条件检查
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

            # 计算新导数
            dphi = float(2 * f_new.T @ (J_new @ p))

            # 曲率条件检查
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
        """Zoom阶段的线搜索"""
        for i in range(10):
            # 二分法选择新的alpha
            alpha = (alpha_lo + alpha_hi) / 2.0

            # 计算新点的函数值
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
