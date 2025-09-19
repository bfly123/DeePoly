import numpy as np
import torch
import time
from typing import Dict, Tuple, Optional
from torch import optim
import os
import argparse
import sys
import matplotlib
import scipy.io
from scipy.interpolate import interp1d

# Check if display is available, otherwise use non-interactive backend
try:
    if os.environ.get('DISPLAY', '') == '':
        matplotlib.use("Agg")  # Use non-interactive backend for headless systems
    else:
        matplotlib.use("TkAgg")  # Use interactive backend for GUI
except:
    matplotlib.use("Agg")  # Fallback to non-interactive backend

import matplotlib.pyplot as plt

# Ensure project modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change relative imports to absolute imports
from src.problem_solvers.time_pde_solver.core.net import TimePDENet

from src.problem_solvers.time_pde_solver.core.fitter import TimePDEFitter
from src.problem_solvers.time_pde_solver.utils.data import TimePDEDataGenerator
from src.problem_solvers.time_pde_solver.utils.visualize import TimePDEVisualizer
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig


class TimePDESolver:
    """
    Optimized time PDE solver
    Supports IMEX-RK time stepping and hybrid neural network-polynomial methods
    """
    
    def __init__(self, config=None, case_dir=None):
        """Initialize solver"""
        self._initialize_config(config, case_dir)
        self._initialize_components()
        self._initialize_solution_tracking()

    def _initialize_config(self, config, case_dir):
        """Initialize configuration"""
        if config is not None:
            self.config = config
        elif case_dir is not None:
            self.config = TimePDEConfig(case_dir=case_dir)
        else:
            raise ValueError("Either config object or case_dir must be provided")

        self.case_dir = case_dir if case_dir else getattr(self.config, "case_dir", None)
        
        # Export configuration file
        if hasattr(self.config, "export_to_json"):
            self.config.export_to_json("config.json")

    def _initialize_components(self):
        """Initialize core components"""
        # Data generator
        self.datagen = TimePDEDataGenerator(self.config)
        
        # Generate training and test data
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")

        # Neural network model
        self.model = TimePDENet(self.config).to(self.config.device)
        
        # Time PDE fitter
        self.fitter = TimePDEFitter(config=self.config, data=self.data_train)

        # Visualizer
        self.visualizer = TimePDEVisualizer(self.config)
        
        # Load reference solution (if specified in config)
        self.reference_solution = self._load_reference_solution()
        
        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)

    def _initialize_solution_tracking(self):
        """Initialize solution tracking"""
        # Loss history still maintained by solver
        self.loss_history = []
        
        # Animation data now managed by visualizer
        self.visualizer.reset_animation_data()

    def _load_reference_solution(self) -> Optional[Dict]:
        """Load reference solution data"""
        if not hasattr(self.config, 'reference_solution') or not self.config.reference_solution:
            return None
            
        ref_path = self.config.reference_solution
        
        # If relative path, make it relative to case directory
        if not os.path.isabs(ref_path):
            case_dir = getattr(self.config, 'case_dir', os.getcwd())
            ref_path = os.path.join(case_dir, ref_path)
        
        # Check corresponding .mat file for .m file
        if ref_path.endswith('.m'):
            mat_file = ref_path.replace('.m', '.mat')
        else:
            mat_file = ref_path
            
        if not os.path.exists(mat_file):
            print(f"Warning: Reference solution file not found: {mat_file}")
            return None
            
        try:
            print(f"Loading reference solution: {mat_file}")
            data = scipy.io.loadmat(mat_file)
            
            # Extract data
            t_ref = data['t'].flatten()
            x_ref = data['x'].flatten()
            u_ref = data['usol']  # shape: (n_space, n_time) from MATLAB

            # Transpose if needed to get (n_time, n_space) format
            if u_ref.shape[0] == len(x_ref) and u_ref.shape[1] == len(t_ref):
                u_ref = u_ref.T  # Transpose to (n_time, n_space)
            
            print(f"  Time points: {len(t_ref)}")
            print(f"  Spatial points: {len(x_ref)}")
            print(f"  Time range: [{t_ref.min():.6f}, {t_ref.max():.6f}]")
            print(f"  Spatial range: [{x_ref.min():.6f}, {x_ref.max():.6f}]")
            
            # Create interpolation functions to get solution at any time
            interpolators = {}
            for i in range(len(x_ref)):
                interpolators[i] = interp1d(t_ref, u_ref[:, i], kind='cubic', 
                                          bounds_error=False, fill_value='extrapolate')
            
            return {
                't_ref': t_ref,
                'x_ref': x_ref, 
                'u_ref': u_ref,
                'interpolators': interpolators,
                'file_path': mat_file
            }
            
        except Exception as e:
            print(f"Error: Cannot read reference solution file {mat_file}: {e}")
            return None
    
    def get_reference_solution_at_time(self, t: float) -> Optional[np.ndarray]:
        """Get reference solution at specified time"""
        if self.reference_solution is None:
            return None
            
        try:
            x_ref = self.reference_solution['x_ref']
            interpolators = self.reference_solution['interpolators']
            
            # Interpolate to get solution at specified time
            u_at_t = np.zeros(len(x_ref))
            for i in range(len(x_ref)):
                u_at_t[i] = interpolators[i](t)
                
            return u_at_t
            
        except Exception as e:
            print(f"Warning: Cannot get reference solution at time t={t}: {e}")
            return None

    def solve(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """Main solve function"""
        print("=== Starting Time PDE Solving ===")
        start_time = time.time()

        # Execute time evolution
        U_final, U_seg_final, model_final, coeffs_final = self.solve_time_evolution()

        # 计算总时间
        total_time = time.time() - start_time
        print(f"Total solving time: {total_time:.2f} seconds")

        # 后处理和可视化
        self._postprocess_results(U_final, U_seg_final, model_final, coeffs_final)

        return U_final, U_seg_final, model_final, coeffs_final

    def solve_time_evolution(
        self,
    ) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """Time evolution solving using IMEX-RK(2,2,2) method"""
        # Initialize time evolution
        it, T, dt, U, U_seg, coeffs = self._initialize_time_evolution()

        # Main time stepping loop
        while T < self.config.T:
            # Compute adaptive time step
            dt = self._compute_adaptive_timestep(it, T, dt, U)

            print(f"Step {it}: T = {T:.6f}, dt = {dt:.6f}")
            
            # Reset linear solver step counter for this time step
            if hasattr(self.fitter, 'solver'):
                self.fitter.solver.reset_step_counter()

            # Train neural network and execute time step
            self._train_neural_network_step(it, dt, U_current=U)

            U, U_seg, coeffs = self.fitter.solve_time_step(
                U, U_seg, dt, coeffs_n=coeffs
            )

            # Update time and iteration
            T += dt
            it += 1

            # 动画数据更新和解监控（传递solver实例以便访问参考解）
            self.visualizer.update_animation_data(it, T, self.data_test, self.model, coeffs, self.fitter, solver=self)
            self._monitor_solution(it, U)

        print(f"Time evolution completed. Final time: T = {T:.6f}")

        # 完成实时动画显示（与standalone solver一致）
        if getattr(self.config, 'realtime_visualization', False):
            self.visualizer.finalize_realtime_animation()

        # 从可视化器获取动画数据统计
        time_history, solution_history = self.visualizer.get_animation_data()
        print(f"Collected {len(time_history)} time steps for animation")

        return U, U_seg, self.model, coeffs

    def _initialize_time_evolution(
        self,
    ) -> Tuple[int, float, float, np.ndarray, list, np.ndarray]:
        """Initialize time evolution parameters and solution"""
        it = 0
        T = 0.0
        dt = self.config.dt

        # Initialize solution values directly
        U, U_seg = self.data_train["U"], self.data_train["U_seg"]

        # Initialize coefficients for time evolution
        coeffs = None

        # Initialize fitter with model for operator precompilation
        self.fitter.fitter_init(self.model)

        # 不在初始化时调用update_animation_data，等到第一个真正的时间步
        print(f"Initialized at T = {T:.6f}, ready for time stepping")

        return it, T, dt, U, U_seg, coeffs

    def _compute_adaptive_timestep(self, it: int, T: float, dt: float, U: np.ndarray) -> float:
        """计算自适应时间步长"""
        base_dt = self.config.dt
        
        # 自适应时间步策略
        #if hasattr(self.config, 'adaptive_dt') and self.config.adaptive_dt:
            # CFL条件限制
         #   if hasattr(self.config, 'cfl_number'):
         #       cfl_dt = self._compute_cfl_timestep(U)
         #       dt = min(base_dt, cfl_dt)
            
            # 稳定性限制
         #   if hasattr(self.fitter, "estimate_stable_dt"):
         #       dt_stable = self.fitter.estimate_stable_dt(U)
         #       dt = min(dt, dt_stable)
         #       
         #   # 解变化率限制
         #   if it > 0 and hasattr(self, '_previous_U'):
         #       dt_change = self._compute_solution_change_limit(U, self._previous_U, dt)
         #       dt = min(dt, dt_change)
        #else:
            # 固定时间步 (it参数预留用于未来自适应功能)
        _ = it  # 暂时未使用，但保留用于未来扩展
        dt = base_dt

        # 首步特殊处理
        #if it == 0 and hasattr(self.config, 'initial_dt_factor'):
        #    dt *= self.config.initial_dt_factor

        # 确保不超过最终时间
        if T + dt > self.config.T:
            dt = self.config.T - T

        # 存储当前解用于下一步比较
        self._previous_U = U.copy()

        return dt
    
    def _compute_cfl_timestep(self, U: np.ndarray) -> float:
        """基于CFL条件计算时间步长"""
        # 估计特征速度
        u_max = np.max(np.abs(U))
        dx_min = np.min(np.diff(self.data_train["x"].flatten()))
        
        cfl_number = getattr(self.config, 'cfl_number', 0.5)
        characteristic_speed = max(u_max, 1e-10)  # 避免除零
        
        dt_cfl = cfl_number * dx_min / characteristic_speed
        return dt_cfl
    
    def _compute_solution_change_limit(self, U_new: np.ndarray, U_old: np.ndarray, dt: float) -> float:
        """基于解变化率限制时间步"""
        rel_change = np.linalg.norm(U_new - U_old) / (np.linalg.norm(U_old) + 1e-12)
        max_change_rate = getattr(self.config, 'max_solution_change_rate', 0.1)
        
        if rel_change > max_change_rate:
            # 减小时间步
            dt_new = dt * max_change_rate / rel_change
            return max(dt_new, dt * 0.5)  # 不超过50%减少
        
        return dt

    # 动画数据更新已移至可视化器的update_animation_data方法

    def _monitor_solution(self, it: int, U: np.ndarray) -> None:
        """Monitor solution progress"""
        if it % 10 == 0:  # Every 10 steps
            print(f"  Solution norm: {np.linalg.norm(U):.6e}")

    def _train_neural_network_step(
        self, it: int, dt: float, U_current: np.ndarray = None
    ) -> None:
        """神经网络训练在当前时间步，支持skip和参数继承"""
        
        # 检查是否需要跳过神经网络训练
        spotter_skip = getattr(self.config, 'spotter_skip', 1)  # 默认每步都训练
        skip_training = (it % spotter_skip != 0) and (it > 0)  # 第一步总是训练
        
        if skip_training:
            print(f"  Skipping neural network training (spotter_skip={spotter_skip}, it={it})")
            # 仍然需要重新初始化fitter，但使用现有模型参数
            self.fitter.fitter_init(self.model)
            return
        
        print(f"  Training neural network (it={it}, inheriting from previous parameters)...")
        
        # 不重新初始化模型参数，保持从上次训练继承
        data_GPU = self.model.prepare_gpu_data(self.data_train, U_current)

        # 从上一次的模型状态开始训练（不重置参数）
        self.model.train()  # 设置为训练模式
        
        # 执行训练，从上次的权重继承
        self.model.train_net(self.data_train, self.model, data_GPU, dt=dt)
        
        # 计算最终损失
        self.model.eval()
        final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
        print(f"  Neural network training completed, loss: {final_loss:.8e}")
        
        # 记录损失历史
        self.loss_history.append(final_loss)
        
        # 重新初始化fitter以使用更新后的模型
        self.fitter.fitter_init(self.model)

        # 可选的可视化（只在训练时绘制）
        if getattr(self.config, 'plot_nn_predictions', False):
            self._plot_neural_network_predictions(it, dt)

    def _plot_neural_network_predictions(self, it: int, dt: float) -> None:
        """Plot neural network predictions as scatter plot"""
        try:
            print("  Creating neural network prediction scatter plot...")

            # Get test data points
            x_test = self.data_test["x"].flatten()
            print(
                f"  Debug: x_test shape: {x_test.shape}, range: [{np.min(x_test):.3f}, {np.max(x_test):.3f}]"
            )

            # Get neural network predictions on test data
            x_tensor = torch.tensor(
                self.data_test["x"], dtype=torch.float64, device=self.config.device
            )
            x_tensor.requires_grad_(True)  # Enable gradients for x
            print(f"  Debug: x_tensor shape: {x_tensor.shape}")

            # Check network parameters first
            total_params = sum(p.numel() for p in self.model.parameters())
            param_norm = sum(p.norm().item() for p in self.model.parameters())
            print(
                f"  Debug: Total parameters: {total_params}, Parameter norm: {param_norm:.6e}"
            )

            # Check input normalization
            print(
                f"  Debug: x_tensor stats - mean: {torch.mean(x_tensor):.6f}, std: {torch.std(x_tensor):.6f}"
            )

            # Don't use no_grad for prediction since we need gradients during forward pass
            self.model.eval()
            features, u_pred = self.model(x_tensor)
            print(f"  Debug: features shape: {features.shape}")
            print(f"  Debug: u_pred shape: {u_pred.shape}")
            print(
                f"  Debug: u_pred range: [{torch.min(u_pred):.6f}, {torch.max(u_pred):.6f}]"
            )
            print(
                f"  Debug: features range: [{torch.min(features):.6f}, {torch.max(features):.6f}]"
            )

            # Check if u_pred is all zeros or very small
            u_pred_abs_max = torch.max(torch.abs(u_pred))
            print(f"  Debug: max absolute value of u_pred: {u_pred_abs_max:.6e}")

            # Check final layer weights
            if hasattr(self.model, "out"):
                out_weight_norm = torch.norm(self.model.out.weight).item()
                print(f"  Debug: Output layer weight norm: {out_weight_norm:.6e}")

            u_pred_np = u_pred.detach().cpu().numpy().flatten()

            # Get target/true values if available
            u_true = self.data_test["u"].flatten() if "u" in self.data_test else None
            if u_true is not None:
                print(
                    f"  Debug: u_true shape: {u_true.shape}, range: [{np.min(u_true):.3f}, {np.max(u_true):.3f}]"
                )

            # Create scatter plot
            plt.figure(figsize=(12, 8))

            # Plot neural network predictions
            plt.scatter(
                x_test,
                u_pred_np,
                c="red",
                alpha=0.7,
                s=30,
                label="NN Predictions",
                marker="o",
            )

            # Plot true values if available
            if u_true is not None:
                plt.scatter(
                    x_test,
                    u_true,
                    c="blue",
                    alpha=0.5,
                    s=20,
                    label="True Values",
                    marker=".",
                )

            # Add labels and formatting
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.title(f"Neural Network Predictions vs True Values (dt={dt:.6f})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save the plot
            results_dir = os.path.join(self.config.results_dir)
            os.makedirs(results_dir, exist_ok=True)

            # Create filename with time step info
            time_history, _ = self.visualizer.get_animation_data()
            if len(time_history) > 0:
                current_time = time_history[-1]
                filename = f"nn_predictions_it_{it:04d}_t_{current_time:.6f}_dt_{dt:.6f}.png"
            else:
                filename = f"nn_predictions_it_{it:04d}_dt_{dt:.6f}.png"

            output_path = os.path.join(results_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()  # Close to free memory

            print(f"  Neural network prediction plot saved to: {output_path}")

            # Print some statistics
            if u_true is not None:
                mse = np.mean((u_pred_np - u_true) ** 2)
                max_error = np.max(np.abs(u_pred_np - u_true))
                print(f"  MSE between NN and true values: {mse:.6e}")
                print(f"  Max absolute error: {max_error:.6e}")

            print(
                f"  NN prediction range: [{np.min(u_pred_np):.6e}, {np.max(u_pred_np):.6e}]"
            )

        except Exception as e:
            print(f"  Warning: Failed to create neural network prediction plot: {e}")

    # ==================== 后处理和可视化方法 ====================

    def _postprocess_results(self, U_final: np.ndarray, U_seg_final: list, 
                           model_final: torch.nn.Module, coeffs_final: np.ndarray):
        """后处理结果 - 使用独立的可视化器"""
        # U_final, U_seg_final 预留用于未来扩展
        _ = U_final, U_seg_final
        print("=== Post-processing and Visualization ===")
        
        # 生成最终解的可视化
        test_predictions, _ = self.fitter.construct(self.data_test, model_final, coeffs_final)
        
        # 绘制最终解 (支持1D/2D自适应)
        final_solution_path = os.path.join(self.config.results_dir, "final_solution.png")
        self.visualizer.plot_solution(self.data_test, test_predictions, final_solution_path)
        
        # 绘制误差分布 (支持1D/2D自适应)
        error_path = os.path.join(self.config.results_dir, "error_distribution.png")
        self.visualizer.plot_error(self.data_test, test_predictions, error_path)
        
        # 从可视化器获取时间演化数据
        time_history, solution_history = self.visualizer.get_animation_data()
        
        if len(time_history) > 1 and len(solution_history) > 1:
            # 使用增强的时间演化总结方法
            evolution_summary_path = os.path.join(self.config.results_dir, "time_evolution_summary.png")
            self.visualizer.plot_time_evolution_summary(time_history, solution_history, evolution_summary_path)
            
            # 生成完整的时间演化GIF动画
            animation_path = os.path.join(self.config.results_dir, "time_evolution.gif")
            self.visualizer.create_time_evolution_gif(time_history, solution_history, self.data_test, animation_path, solver=self)
        else:
            print("Warning: Insufficient time evolution data for visualization")

        # 生成损失曲线（使用可视化器方法）
        if self.loss_history:
            loss_path = os.path.join(self.config.results_dir, "loss_history.png")
            self.visualizer.plot_loss_history(self.loss_history, loss_path)
        
        # 生成参考解对比分析（如果有参考解）
        if self.reference_solution is not None:
            self._generate_reference_comparison_analysis(model_final, coeffs_final)
            
        print(f"=== Visualization results saved to: {self.config.results_dir} ===")

    def _generate_reference_comparison_analysis(self, model: torch.nn.Module, coeffs: np.ndarray):
        """生成参考解对比分析"""
        print("=== Generating Reference Solution Comparison Analysis ===")
        
        try:
            # 在最终时刻获取数值解
            T_final = getattr(self.config, 'T', 0.2)
            numerical_solution, _ = self.fitter.construct(self.data_test, model, coeffs)
            
            # 获取最终时刻的参考解
            reference_solution = self.get_reference_solution_at_time(T_final)
            
            if reference_solution is None:
                print("Warning: Cannot obtain reference solution for comparison")
                return
                
            # 获取空间坐标
            if 'x_segments' in self.data_test:
                x_coords = np.vstack(self.data_test['x_segments'])
            elif 'x' in self.data_test:
                x_coords = self.data_test['x']
            else:
                print("Warning: Cannot obtain spatial coordinates")
                return
            
            x_test = x_coords[:, 0] if x_coords.ndim > 1 else x_coords
            
            # 插值参考解到测试网格
            if len(reference_solution) != len(numerical_solution):
                from scipy.interpolate import interp1d
                x_ref = self.reference_solution['x_ref']
                interp_func = interp1d(x_ref, reference_solution, kind='cubic', 
                                     bounds_error=False, fill_value='extrapolate')
                reference_interp = interp_func(x_test)
            else:
                reference_interp = reference_solution
                
            # 计算各种误差指标
            abs_error = np.abs(numerical_solution.flatten() - reference_interp)
            rel_error = abs_error / (np.abs(reference_interp) + 1e-12)
            
            max_error = np.max(abs_error)
            l2_error = np.sqrt(np.mean(abs_error**2))
            l_inf_error = np.max(abs_error)
            mean_rel_error = np.mean(rel_error)
            
            # 保存对比分析图
            comparison_path = os.path.join(self.config.results_dir, "reference_comparison.png")
            self._plot_reference_comparison(x_test, numerical_solution.flatten(), reference_interp, 
                                          abs_error, comparison_path)
            
            # 生成误差报告
            report_path = os.path.join(self.config.results_dir, "error_analysis_report.txt")
            self._generate_error_report(T_final, max_error, l2_error, l_inf_error, 
                                      mean_rel_error, report_path)
            
            print(f"Reference solution comparison analysis completed:")
            print(f"  最大误差: {max_error:.2e}")
            print(f"  L2误差: {l2_error:.2e}")
            print(f"  L∞误差: {l_inf_error:.2e}")
            print(f"  平均相对误差: {mean_rel_error:.2e}")
            
        except Exception as e:
            print(f"参考解对比分析失败: {e}")
    
    def _plot_reference_comparison(self, x: np.ndarray, u_numerical: np.ndarray, 
                                 u_reference: np.ndarray, abs_error: np.ndarray, save_path: str):
        """绘制参考解对比图"""
        fig = plt.figure(figsize=(15, 10))
        
        # 解对比
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(x, u_reference, 'b-', linewidth=2, label='Reference Solution', alpha=0.8)
        ax1.plot(x, u_numerical, 'r--', linewidth=2, label='Numerical Solution', alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u')
        ax1.set_title('Solution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绝对误差
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(x, abs_error, 'r-', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('|Error|')
        ax2.set_title('Absolute Error Distribution')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 相对误差
        ax3 = fig.add_subplot(2, 2, 3)
        rel_error = abs_error / (np.abs(u_reference) + 1e-12)
        ax3.plot(x, rel_error, 'g-', linewidth=2)
        ax3.set_xlabel('x')
        ax3.set_ylabel('Relative Error')
        ax3.set_title('Relative Error Distribution')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 误差统计
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(np.log10(abs_error + 1e-16), bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('log10(Absolute Error)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution Histogram')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_error_report(self, T_final: float, max_error: float, l2_error: float, 
                             l_inf_error: float, mean_rel_error: float, report_path: str):
        """生成误差分析报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Allen-Cahn方程数值解误差分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"分析时刻: T = {T_final:.6f}\n")
            f.write(f"参考解: {self.reference_solution['file_path']}\n")
            f.write(f"参考解精度: {len(self.reference_solution['x_ref'])} 空间点, {len(self.reference_solution['t_ref'])} 时间点\n\n")
            
            f.write("误差指标:\n")
            f.write("-" * 30 + "\n")
            f.write(f"最大绝对误差 (L∞): {max_error:.6e}\n")
            f.write(f"L2误差:              {l2_error:.6e}\n")
            f.write(f"平均相对误差:        {mean_rel_error:.6e}\n\n")
            
            f.write("配置信息:\n")
            f.write("-" * 30 + "\n")
            f.write(f"神经网络层数: {len(self.config.hidden_dims)}\n")
            f.write(f"隐藏层维度: {self.config.hidden_dims}\n")
            f.write(f"多项式段数: {self.config.n_segments}\n")
            f.write(f"多项式阶数: {self.config.poly_degree}\n")
            f.write(f"训练点数: {self.config.points_domain}\n")
            f.write(f"测试点数: {self.config.points_domain_test}\n")
            f.write(f"时间步长: {self.config.dt}\n")
            f.write(f"神经网络跳过间隔: {getattr(self.config, 'spotter_skip', 1)}\n\n")
            
            # 收敛信息
            if self.loss_history:
                f.write("收敛信息:\n")
                f.write("-" * 30 + "\n")
                f.write(f"初始损失: {self.loss_history[0]:.6e}\n")
                f.write(f"最终损失: {self.loss_history[-1]:.6e}\n")
                f.write(f"损失下降比: {self.loss_history[0]/self.loss_history[-1]:.2e}\n")

    # 移除冗余方法，使用可视化器的功能

    # 移除冗余方法，使用可视化器的plot_loss_history方法

    # ==================== 便利接口 ====================

    def get_solution_at_time(self, T_target: float) -> Tuple[np.ndarray, float]:
        """获取指定时间的解"""
        time_history, solution_history = self.visualizer.get_animation_data()
        
        if not time_history:
            raise RuntimeError("需要先运行solve()方法")
            
        # 找到最接近的时间点
        time_array = np.array(time_history)
        idx = np.argmin(np.abs(time_array - T_target))
        
        return solution_history[idx], time_array[idx]

    def get_convergence_info(self) -> Dict:
        """获取收敛信息"""
        if not self.loss_history:
            return {"error": "没有可用的收敛数据"}
            
        return {
            "final_loss": self.loss_history[-1],
            "initial_loss": self.loss_history[0],
            "reduction_ratio": self.loss_history[0] / self.loss_history[-1],
            "total_steps": len(self.loss_history),
            "average_loss": np.mean(self.loss_history)
        }

    def export_solution_data(self, filepath: str):
        """导出解数据"""
        # 从可视化器获取时间演化数据
        time_history, solution_history = self.visualizer.get_animation_data()
        
        export_data = {
            'config': self.config.__dict__,
            'time_history': time_history,
            'solution_history': solution_history,
            'loss_history': self.loss_history,
            'x_coords': self.data_test["x"].flatten() if "x" in self.data_test else np.array([])
        }
        
        np.savez_compressed(filepath, **export_data)
        print(f"解数据导出至: {filepath}")


# ==================== 便利函数 ====================

def solve_time_pde(case_dir: str) -> TimePDESolver:
    """便利函数：直接求解时间PDE"""
    solver = TimePDESolver(case_dir=case_dir)
    solver.solve()
    return solver
