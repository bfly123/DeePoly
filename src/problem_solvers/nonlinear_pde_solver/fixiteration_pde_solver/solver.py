"""
FixIterationPDESolver - Fixed-Point Iteration Solver

Solves steady-state PDEs using fixed-point iteration.
"""

import numpy as np
import os
from typing import Optional, Dict, Any

from ..solver import NonlinearPDESolver, IterationState
from .core.fitter import FixIterationPDEFitter
from .core.net import FixIterationPDENet
from .utils.config import FixIterationPDEConfig
from .utils.data import FixIterationPDEDataGenerator
from .utils.visualize import FixIterationPDEVisualizer


class FixIterationPDESolver(NonlinearPDESolver):
    """Fixed-point iteration solver for steady-state PDEs"""

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Solve using fixed-point iteration with DNN training at each step

        Args:
            U_init: Initial guess (optional)

        Returns:
            Dict with solution, coefficients, and convergence info
        """
        print("\n" + "="*70)
        print("Fixed-Point Iteration Solver with DNN Training")
        print("="*70)
        print(f"Domain: {self.config.x_domain}")
        print(f"Segments: {self.config.n_segments}")
        print(f"Polynomial degree: {self.config.poly_degree}")
        print("="*70 + "\n")

        # Initialize solution
        if U_init is None:
            U_init = self._initialize_from_config()

        # Initialize fitter
        self.fitter.fitter_init(self.model)

        # Solve with DNN training managed in solver layer
        U_final, coeffs_final, conv_info = self._solve_with_dnn_training(U_init)

        # Post-processing (always visualize, even if not converged)
        self._postprocess_results(U_final, coeffs_final, conv_info)

        return {
            "solution": U_final,
            "coefficients": coeffs_final,
            "convergence_info": conv_info
        }

    def _solve_with_dnn_training(self, U_init: np.ndarray) -> tuple:
        """Fixed-point iteration: train network once, iterate coefficients

        Phase 1 (Initialization): Train network → get fixed features Φ(x)
        Phase 2 (Iteration): Fix Φ(x), iterate coefficients β solving:
            [L1 + L2⊙F(Φβ)] β = S - N(Φβ)

        Args:
            U_init: Initial guess

        Returns:
            (U_final, coeffs_final, convergence_info)
        """
        import time

        # Get iteration parameters from fitter
        max_iterations = self.fitter.max_iterations
        convergence_tol = self.fitter.convergence_tol
        print_interval = getattr(self.fitter, 'print_interval', 10)

        # Initialize iteration
        U_current = U_init.copy()
        coeffs_current = None
        residual_history = []

        # ============ 初始化阶段：训练网络一次 ============
        print("Initializing network features...")
        dt_virtual = 0.01
        self._train_neural_network_step(0, dt_virtual, U_current)
        print("✓ Network features initialized\n")

        # 获取网络预测作为Sniper迭代的初始值
        U_current = self._get_network_prediction()
        print(f"Network prediction obtained: ||U_net|| = {np.linalg.norm(U_current):.6e}")
        print(f"  U_net range: [{np.min(U_current):.6e}, {np.max(U_current):.6e}]\n")

        # 保存网络预测与参考解对比图
        self._save_network_comparison()

        # ============ 迭代阶段：固定特征基，迭代系数 ============
        print("=" * 70)
        print("Fixed-Point Iteration (Sniper coefficients)")
        print("=" * 70)
        print(f"Max iterations: {max_iterations}")
        print(f"Convergence tolerance: {convergence_tol:.2e}")
        print(f"Starting U_current: ||U|| = {np.linalg.norm(U_current):.6e}")
        print(f"  Range: [{np.min(U_current):.6f}, {np.max(U_current):.6f}]")
        print("=" * 70)

        start_time = time.time()

        for iteration in range(max_iterations):
            # 线性求解（用当前迭代值构建Jacobian）
            print(f"\nIteration {iteration+1}: Calling fitter.fit() with U_current...")
            print(f"  Input U_current: ||U|| = {np.linalg.norm(U_current):.6e}")
            coeffs_new = self.fitter.fit(U_current=U_current)

            # 重建解
            U_new, _ = self.fitter.construct(self.data_train, self.model, coeffs_new)

            # Compute residual
            residual = np.linalg.norm(U_new - U_current) / (np.linalg.norm(U_new) + 1e-10)
            residual_history.append(residual)

            # Print progress
            if iteration % print_interval == 0 or residual < convergence_tol:
                print(f"Iter {iteration+1:4d} | Residual: {residual:.6e} | ||U||: {np.linalg.norm(U_new):.6e}")

            # Check convergence
            if residual < convergence_tol:
                elapsed = time.time() - start_time
                print("\n" + "=" * 70)
                print(f"✓ Converged at iteration {iteration+1}!")
                print(f"  Final residual: {residual:.6e}")
                print(f"  Total time: {elapsed:.2f}s")
                print("=" * 70)

                # Store residual history in fitter for visualization compatibility
                self.fitter.residual_history = residual_history

                return U_new, coeffs_new, {
                    "converged": True,
                    "iterations": iteration + 1,
                    "final_residual": residual,
                    "elapsed_time": elapsed,
                    "residual_history": residual_history
                }

            # Update for next iteration
            U_current = U_new.copy()
            coeffs_current = coeffs_new.copy()

        # Maximum iterations reached
        elapsed = time.time() - start_time
        print(f"\n✗ Maximum iterations ({max_iterations}) reached")
        print(f"  Final residual: {residual:.6e}")

        # Store residual history in fitter for visualization compatibility
        self.fitter.residual_history = residual_history

        return U_current, coeffs_current, {
            "converged": False,
            "reason": "max_iterations",
            "iterations": max_iterations,
            "final_residual": residual,
            "elapsed_time": elapsed,
            "residual_history": residual_history
        }

    def _get_network_prediction(self) -> np.ndarray:
        """从训练后的网络提取预测值

        FixIteration特有：网络既提供特征基Φ(x)又提供初始猜测U_net(x)

        Returns:
            网络预测的解 (n_points, n_eqs)
        """
        import torch

        self.model.eval()
        with torch.no_grad():
            x_train_tensor = torch.tensor(
                self.data_train['x'],
                dtype=torch.float64,
                device=self.config.device
            )
            _, U_pred = self.model(x_train_tensor)
            U_network = U_pred.cpu().numpy()

        return U_network

    def _postprocess_results(self, U_final, coeffs_final, conv_info):
        """Post-process and visualize results"""
        U_final_cpu = self._to_numpy_cpu(U_final)
        coeffs_cpu = self._to_numpy_cpu(coeffs_final)

        # Interpolate to test grid
        U_test = self._interpolate_to_test_grid(U_final_cpu)

        # Save and visualize
        results_dir = self.config.results_dir
        self.visualizer.save_convergence_history(conv_info, results_dir)
        self.visualizer.plot_final_solution(U_test, self.data_test,
                                           os.path.join(results_dir, "final_solution.png"))
        self.visualizer.plot_residual_history(os.path.join(results_dir, "residual_history.png"))

        # 保存Sniper结果与参考解对比图
        self._save_sniper_comparison(U_test)

        print(f"✓ Results saved to: {results_dir}")

    # ==================== 时间步控制方法（FixIteration特定） ====================

    def get_solver_time_info(self) -> Dict[str, Any]:
        """获取迭代信息 - FixIteration特定覆盖

        Returns:
            包含固定点迭代信息的字典
        """
        info = super().get_solver_time_info()
        info.update({
            "solver_type": "fixed_point_iteration",
            "has_time_stepping": False,
            "convergence_based": True,
            "max_iterations": getattr(self.fitter, 'max_iterations', 'N/A'),
            "convergence_tolerance": getattr(self.fitter, 'convergence_tol', 'N/A')
        })
        return info

    # Factory methods
    def _create_config(self, case_dir: str):
        return FixIterationPDEConfig(case_dir=case_dir)

    def _create_data_generator(self):
        return FixIterationPDEDataGenerator(self.config)

    def _create_model(self):
        return FixIterationPDENet.model_init(self.config)

    def _create_fitter(self, time_scheme: Optional[str]):
        return FixIterationPDEFitter(config=self.config, data=self.data_train)

    def _create_visualizer(self):
        return FixIterationPDEVisualizer(self.config)

    def _get_default_time_scheme(self) -> Optional[str]:
        return None

    def _process_reference_solution(self, data: Dict, x_ref: np.ndarray, mat_file: str) -> Dict:
        """Process reference solution data for 2D steady-state problems

        For 2D problems (x, y) where the mat file contains (t, x) data,
        we treat t as the second spatial dimension (y).
        """
        # Extract solution
        u_ref = data.get('usol', data.get('u', None))
        if u_ref is None:
            return None

        # Extract second dimension coordinate (t treated as y)
        y_ref = None
        if 't' in data:
            y_ref = data['t'].flatten()
        elif 'y' in data:
            y_ref = data['y'].flatten()

        # For 2D case, keep the structure
        if u_ref.ndim == 2:
            # Flatten for interpolation but keep both coordinates
            u_ref_flat = u_ref.flatten()

            ref_dict = {
                'x_ref': x_ref.flatten(),
                'u_ref': u_ref_flat,
                'file_path': mat_file
            }

            # Add second dimension if available
            if y_ref is not None:
                ref_dict['y_ref'] = y_ref
                ref_dict['y_label'] = 't'  # Label for visualization

            return ref_dict
        else:
            # 1D case
            return {
                'x_ref': x_ref.flatten(),
                'u_ref': u_ref.flatten(),
                'file_path': mat_file
            }

    def _get_spotter_net_class(self):
        """Get SpotterNet class for initialization"""
        from ..pseudotime_pde_solver.core.spotter_init import SpotterNet
        return SpotterNet

    def _save_network_comparison(self):
        """保存网络预测与参考解的对比图

        生成三个子图：
        1. 网络预测的解（在测试网格上）
        2. 参考解（高分辨率）
        3. 逐点误差（对数刻度）
        """
        if not hasattr(self, 'reference_solution') or self.reference_solution is None:
            print("  ⚠ No reference solution available, skipping network comparison")
            return

        print("\n" + "=" * 70)
        print("Saving Network Prediction Comparison")
        print("=" * 70)

        # 获取网络预测
        U_network_train = self._get_network_prediction()

        # 插值到测试网格
        U_network_test = self._interpolate_to_test_grid(U_network_train)

        # 保存对比图（三个子图：数值解、参考解、残差）
        results_dir = self.config.results_dir
        save_path = os.path.join(results_dir, "network_vs_reference.png")

        self.visualizer.plot_reference_comparison(
            U_network_test,
            self.data_test,
            self.reference_solution,
            save_path
        )
        print("=" * 70 + "\n")

    def _save_sniper_comparison(self, U_sniper_test):
        """保存Sniper结果与参考解的对比图

        生成三个子图：
        1. Sniper优化后的解（在测试网格上）
        2. 参考解（高分辨率）
        3. 逐点误差（对数刻度）

        Args:
            U_sniper_test: Sniper在测试集上的解
        """
        if not hasattr(self, 'reference_solution') or self.reference_solution is None:
            print("  ⚠ No reference solution available, skipping Sniper comparison")
            return

        print("\n" + "=" * 70)
        print("Saving Sniper Results Comparison")
        print("=" * 70)

        # 保存对比图（三个子图：数值解、参考解、残差）
        results_dir = self.config.results_dir
        save_path = os.path.join(results_dir, "sniper_vs_reference.png")

        self.visualizer.plot_reference_comparison(
            U_sniper_test,
            self.data_test,
            self.reference_solution,
            save_path
        )
        print("=" * 70)
