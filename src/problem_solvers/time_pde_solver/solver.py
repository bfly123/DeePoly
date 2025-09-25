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
        
        # Time PDE fitter with specified time scheme
        time_scheme = getattr(self.config, 'time_scheme', 'imex_rk_222')
        self.fitter = TimePDEFitter(config=self.config, data=self.data_train, time_scheme=time_scheme)

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
                U, U_seg, dt, coeffs_n=coeffs, step=it, current_time=T
            )

            # Update time and iteration
            T += dt
            it += 1

            # 动画数据更新和解监控（传递solver实例以便访问参考解，直接使用时间步输出的U）
            self.visualizer.update_animation_data(it, T, self.data_train, self.model, coeffs, self.fitter, solver=self, U_direct=U, U_seg_direct=U_seg)
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
            # 固定时间步
        dt = base_dt

        # 根据时间格式进行首步特殊处理
        time_scheme = getattr(self.config, 'time_scheme', 'imex_rk_222')
        if time_scheme == "onestep_predictor" and it == 0:
            # OneStep Predictor: 首步时间缩减到1/10
            dt *= 0.1
            print(f"  OneStep Predictor: First step dt reduction, dt = {dt:.6f}")
        elif it == 0 and hasattr(self.config, 'initial_dt_factor'):
            # 其他格式的首步处理
            dt *= self.config.initial_dt_factor

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
        """后处理结果 - 专注于train数据与参考解对比"""
        print("=== Post-processing and Visualization ===")

        # 设置matplotlib样式以获得更专业的图形
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.facecolor': 'white'
        })

        # 从可视化器获取时间演化数据（使用train数据）
        time_history, solution_history = self.visualizer.get_animation_data()

        if len(time_history) == 0 or len(solution_history) == 0:
            print("Warning: No time evolution data available for analysis")
            return

        # 1. 最终时刻train结果与参考解对比
        print("1. 分析最终时刻解与参考解的对比...")
        self._plot_final_time_comparison(time_history, solution_history)

        # 2. t-x二维图：train解、参考解和逐点误差
        print("2. 生成时空二维图...")
        self._plot_spacetime_comparison(time_history, solution_history)

        # 3. 统计所有时刻的平均误差
        print("3. 统计所有时刻的误差...")
        self._compute_overall_error_statistics(time_history, solution_history)

        # 4. 导出数据供MATLAB使用
        print("4. 导出数据供MATLAB分析...")
        self._export_data_for_matlab(time_history, solution_history)

        print(f"=== Analysis results saved to: {self.config.results_dir} ===")
        print("=== To generate MATLAB plots, run: matlab -batch 'plot_comparison' in the reference_data directory ===")

    def _plot_final_time_comparison(self, time_history, solution_history):
        """1. 最终时刻train结果与参考解对比和逐点误差"""
        if not self.reference_solution:
            print("  No reference solution available for comparison")
            return

        # 获取最终时刻
        T_final = time_history[-1]
        U_final = solution_history[-1]

        # 获取参考解
        u_ref_final = self.get_reference_solution_at_time(T_final)
        if u_ref_final is None:
            print("  Cannot get reference solution at final time")
            return

        # 获取空间坐标
        x_coords = self.data_train.get('x', self.data_train.get('x_segments', []))
        if isinstance(x_coords, list):
            x_plot = np.vstack(x_coords) if x_coords else np.array([[0]])
        else:
            x_plot = x_coords
        x_flat = x_plot[:, 0] if x_plot.ndim > 1 else x_plot

        # 插值参考解到train网格
        from scipy.interpolate import interp1d
        x_ref = self.reference_solution['x_ref']
        interp_func = interp1d(x_ref, u_ref_final, kind='cubic', bounds_error=False, fill_value='extrapolate')
        u_ref_interp = interp_func(x_flat)

        U_flat = U_final.flatten()

        # 计算逐点误差
        pointwise_error = np.abs(U_flat - u_ref_interp)

        # 绘图 - 使用更专业的样式
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 上图：解对比
        # Debug: 打印参考解信息
        print(f"  DEBUG - Reference solution: x_ref.shape={x_ref.shape}, u_ref_final.shape={u_ref_final.shape}")
        print(f"  DEBUG - x_ref range: [{x_ref.min():.6f}, {x_ref.max():.6f}]")
        print(f"  DEBUG - u_ref_final range: [{u_ref_final.min():.6f}, {u_ref_final.max():.6f}]")
        print(f"  DEBUG - x_ref is sorted: {np.all(np.diff(x_ref) >= 0)}")

        # 先绘制参考解（连续线）作为背景
        ax1.plot(x_ref, u_ref_final, 'b-', linewidth=2.5, alpha=0.8,
                label='Reference Solution', zorder=1)
        # 再绘制train数据（散点）突出显示
        scatter1 = ax1.scatter(x_flat, U_flat, c='red', s=35, alpha=0.9,
                              edgecolors='darkred', linewidths=0.5,
                              label='Train Solution', zorder=2)

        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('u', fontsize=12)
        ax1.set_title(f'Final Time Comparison (T = {T_final:.4f})',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)

        # 设置坐标轴范围
        u_min_plot = min(np.min(U_flat), np.min(u_ref_final))
        u_max_plot = max(np.max(U_flat), np.max(u_ref_final))
        u_margin = (u_max_plot - u_min_plot) * 0.05
        ax1.set_ylim([u_min_plot - u_margin, u_max_plot + u_margin])

        # 下图：逐点误差 - 使用单一颜色
        scatter2 = ax2.scatter(x_flat, pointwise_error, c='red', s=25, alpha=0.8,
                              edgecolors='darkred', linewidths=0.3)
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('|Error|', fontsize=12)
        ax2.set_title('Pointwise Error at Final Time (Log Scale)',
                     fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        # 移除颜色条，不再需要

        # 添加误差统计信息
        max_error = np.max(pointwise_error)
        l2_error = np.sqrt(np.mean(pointwise_error**2))
        ax2.text(0.02, 0.98, f'Max Error: {max_error:.2e}\\nL2 Error: {l2_error:.2e}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        final_comparison_path = os.path.join(self.config.results_dir, "final_time_comparison.png")
        plt.savefig(final_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Final time comparison saved to: {final_comparison_path}")
        print(f"  Final time error - Max: {max_error:.2e}, L2: {l2_error:.2e}")

    def _plot_spacetime_comparison(self, time_history, solution_history):
        """2. t-x二维图：train解、参考解和逐点误差"""
        if not self.reference_solution:
            print("  No reference solution available for spacetime plot")
            return

        # 获取空间坐标
        x_coords = self.data_train.get('x', self.data_train.get('x_segments', []))
        if isinstance(x_coords, list):
            x_plot = np.vstack(x_coords) if x_coords else np.array([[0]])
        else:
            x_plot = x_coords
        x_flat = x_plot[:, 0] if x_plot.ndim > 1 else x_plot

        # 构建时空网格数据
        nt = len(time_history)
        nx = len(x_flat)

        U_spacetime = np.zeros((nt, nx))
        U_ref_spacetime = np.zeros((nt, nx))

        x_ref = self.reference_solution['x_ref']

        # 填充数据
        for i, (t, U_t) in enumerate(zip(time_history, solution_history)):
            # Train解
            U_spacetime[i, :] = U_t.flatten()

            # 参考解（插值到train网格）
            u_ref_t = self.get_reference_solution_at_time(t)
            if u_ref_t is not None:
                from scipy.interpolate import interp1d
                interp_func = interp1d(x_ref, u_ref_t, kind='cubic', bounds_error=False, fill_value='extrapolate')
                U_ref_spacetime[i, :] = interp_func(x_flat)
            else:
                U_ref_spacetime[i, :] = 0

        # 计算逐点误差
        error_spacetime = np.abs(U_spacetime - U_ref_spacetime)

        # 创建网格
        T_grid, X_grid = np.meshgrid(time_history, x_flat, indexing='ij')

        # 绘制三个子图 - 使用更好的可视化方法
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 设置全局颜色范围以便对比
        u_min = min(np.min(U_spacetime), np.min(U_ref_spacetime))
        u_max = max(np.max(U_spacetime), np.max(U_ref_spacetime))

        # 子图1: Train解 - 使用scatter显示随机采样点
        t_scatter = []
        x_scatter = []
        u_scatter = []
        for i, t in enumerate(time_history):
            for j, x in enumerate(x_flat):
                t_scatter.append(t)
                x_scatter.append(x)
                u_scatter.append(U_spacetime[i, j])

        scatter1 = axes[0].scatter(t_scatter, x_scatter, c=u_scatter, cmap='RdBu_r',
                                  s=8, alpha=0.8, vmin=u_min, vmax=u_max, edgecolors='none')
        axes[0].set_xlabel('t', fontsize=12)
        axes[0].set_ylabel('x', fontsize=12)
        axes[0].set_title('Train Solution (Random Sampling)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label('u', fontsize=11)

        # 子图2: 参考解 - 使用pcolormesh模拟MATLAB的pcolor效果
        # 创建高分辨率参考解网格
        t_ref_grid = self.reference_solution['t_ref']
        x_ref_grid = self.reference_solution['x_ref']
        u_ref_grid = self.reference_solution['u_ref']

        # 使用pcolormesh获得更好的效果（类似MATLAB的pcolor）
        T_ref_mesh, X_ref_mesh = np.meshgrid(t_ref_grid, x_ref_grid)
        im2 = axes[1].pcolormesh(T_ref_mesh, X_ref_mesh, u_ref_grid.T, cmap='RdBu_r',
                                shading='gouraud', vmin=u_min, vmax=u_max)
        axes[1].set_xlabel('t', fontsize=12)
        axes[1].set_ylabel('x', fontsize=12)  # 所有图现在都是统一方向：x轴时间，y轴空间
        axes[1].set_title('Reference Solution (High-Res)', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('u', fontsize=11)

        # 子图3: 逐点误差 - 使用scatter显示
        error_scatter = []
        for i, t in enumerate(time_history):
            for j, x in enumerate(x_flat):
                error_scatter.append(error_spacetime[i, j])

        # 改进误差可视化 - 使用更好的颜色映射和范围控制
        from matplotlib.colors import LogNorm

        # 计算误差统计信息以优化颜色范围
        error_array = np.array(error_scatter)
        error_nonzero = np.maximum(error_array, 1e-12)  # 避免log(0)

        # 计算合理的颜色范围（去除极端值影响）
        error_p05 = np.percentile(error_nonzero, 5)   # 5%分位数
        error_p95 = np.percentile(error_nonzero, 95)  # 95%分位数
        error_median = np.median(error_nonzero)

        # 设置颜色范围，突出中等误差的变化
        vmin_error = max(error_p05, 1e-10)
        vmax_error = min(error_p95 * 5, np.max(error_nonzero))  # 扩展上限以显示高误差区域

        print(f"  Error colormap range: [{vmin_error:.2e}, {vmax_error:.2e}], median: {error_median:.2e}")

        # 使用更具对比度的颜色映射
        scatter3 = axes[2].scatter(t_scatter, x_scatter, c=error_nonzero,
                                  cmap='plasma',  # 使用plasma colormap获得更好的对比度
                                  s=10, alpha=0.9,
                                  norm=LogNorm(vmin=vmin_error, vmax=vmax_error),
                                  edgecolors='none')
        axes[2].set_xlabel('t', fontsize=12)
        axes[2].set_ylabel('x', fontsize=12)
        axes[2].set_title('Pointwise Error (Log Scale)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        cbar3 = plt.colorbar(scatter3, ax=axes[2])
        cbar3.set_label('|Error|', fontsize=11)

        plt.tight_layout()
        spacetime_path = os.path.join(self.config.results_dir, "spacetime_comparison.png")
        plt.savefig(spacetime_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Spacetime comparison saved to: {spacetime_path}")

    def _compute_overall_error_statistics(self, time_history, solution_history):
        """3. 统计所有时刻的平均L2误差和L∞误差"""
        if not self.reference_solution:
            print("  No reference solution available for error statistics")
            return

        # 获取空间坐标
        x_coords = self.data_train.get('x', self.data_train.get('x_segments', []))
        if isinstance(x_coords, list):
            x_plot = np.vstack(x_coords) if x_coords else np.array([[0]])
        else:
            x_plot = x_coords
        x_flat = x_plot[:, 0] if x_plot.ndim > 1 else x_plot
        x_ref = self.reference_solution['x_ref']

        l2_errors = []
        linf_errors = []

        # 计算每个时刻的误差
        for t, U_t in zip(time_history, solution_history):
            # 获取参考解
            u_ref_t = self.get_reference_solution_at_time(t)
            if u_ref_t is None:
                continue

            # 插值参考解到train网格
            from scipy.interpolate import interp1d
            interp_func = interp1d(x_ref, u_ref_t, kind='cubic', bounds_error=False, fill_value='extrapolate')
            u_ref_interp = interp_func(x_flat)

            # 计算误差
            U_flat = U_t.flatten()
            error = np.abs(U_flat - u_ref_interp)

            l2_error = np.sqrt(np.mean(error**2))
            linf_error = np.max(error)

            l2_errors.append(l2_error)
            linf_errors.append(linf_error)

        if len(l2_errors) == 0:
            print("  No valid time points for error calculation")
            return

        # 计算时间平均误差
        mean_l2_error = np.mean(l2_errors)
        mean_linf_error = np.mean(linf_errors)
        max_l2_error = np.max(l2_errors)
        max_linf_error = np.max(linf_errors)

        # 保存误差统计报告
        report_path = os.path.join(self.config.results_dir, "error_statistics_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\\n")
            f.write("Train Data Error Statistics Report\\n")
            f.write("=" * 60 + "\\n\\n")

            f.write(f"Time range: [{time_history[0]:.6f}, {time_history[-1]:.6f}]\\n")
            f.write(f"Total time steps analyzed: {len(l2_errors)}\\n")
            f.write(f"Spatial points: {len(x_flat)}\\n\\n")

            f.write("Time-averaged Errors:\\n")
            f.write("-" * 30 + "\\n")
            f.write(f"Mean L2 Error:   {mean_l2_error:.6e}\\n")
            f.write(f"Mean L∞ Error:   {mean_linf_error:.6e}\\n\\n")

            f.write("Maximum Errors (across all times):\\n")
            f.write("-" * 30 + "\\n")
            f.write(f"Max L2 Error:    {max_l2_error:.6e}\\n")
            f.write(f"Max L∞ Error:    {max_linf_error:.6e}\\n\\n")

            f.write("Configuration:\\n")
            f.write("-" * 30 + "\\n")
            f.write(f"Time scheme:     {getattr(self.config, 'time_scheme', 'N/A')}\\n")
            f.write(f"Time step:       {self.config.dt}\\n")
            f.write(f"Neural network:  {self.config.hidden_dims}\\n")
            f.write(f"Segments:        {self.config.n_segments}\\n")
            f.write(f"Polynomial deg:  {self.config.poly_degree}\\n")

        # 绘制误差随时间变化图 - 更专业的样式
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # L2误差演化 - 使用单一颜色
        scatter_l2 = ax1.scatter(time_history[:len(l2_errors)], l2_errors,
                                c='blue', s=40, alpha=0.8,
                                edgecolors='navy', linewidths=0.5, label='L2 Error')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('L2 Error (log scale)', fontsize=12)
        ax1.set_title('L2 Error Evolution (Train Data Points)',
                     fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # 添加趋势线
        if len(l2_errors) > 2:
            z = np.polyfit(time_history[:len(l2_errors)], np.log10(l2_errors), 1)
            p = np.poly1d(z)
            ax1.plot(time_history[:len(l2_errors)], 10**p(time_history[:len(l2_errors)]),
                    'b--', alpha=0.5, linewidth=1.5, label='Trend')

        # L∞误差演化 - 使用单一颜色
        scatter_linf = ax2.scatter(time_history[:len(linf_errors)], linf_errors,
                                  c='red', s=40, alpha=0.8,
                                  edgecolors='darkred', linewidths=0.5, label='L∞ Error')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('L∞ Error (log scale)', fontsize=12)
        ax2.set_title('L∞ Error Evolution (Train Data Points)',
                     fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        # 添加趋势线
        if len(linf_errors) > 2:
            z = np.polyfit(time_history[:len(linf_errors)], np.log10(linf_errors), 1)
            p = np.poly1d(z)
            ax2.plot(time_history[:len(linf_errors)], 10**p(time_history[:len(linf_errors)]),
                    'r--', alpha=0.5, linewidth=1.5, label='Trend')

        # 添加图例
        ax1.legend(fontsize=10)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        error_evolution_path = os.path.join(self.config.results_dir, "error_evolution.png")
        plt.savefig(error_evolution_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Error statistics saved to: {report_path}")
        print(f"  Error evolution plot saved to: {error_evolution_path}")
        print("  Overall Error Statistics:")
        print(f"    Time-averaged L2 Error:  {mean_l2_error:.6e}")
        print(f"    Time-averaged L∞ Error:  {mean_linf_error:.6e}")
        print(f"    Maximum L2 Error:        {max_l2_error:.6e}")
        print(f"    Maximum L∞ Error:        {max_linf_error:.6e}")

    def _export_data_for_matlab(self, time_history, solution_history):
        """导出train数据供MATLAB分析使用"""
        try:
            import scipy.io

            # 获取空间坐标
            x_coords = self.data_train.get('x', self.data_train.get('x_segments', []))
            if isinstance(x_coords, list):
                x_plot = np.vstack(x_coords) if x_coords else np.array([[0]])
            else:
                x_plot = x_coords
            x_flat = x_plot[:, 0] if x_plot.ndim > 1 else x_plot

            # 构建解矩阵 (nx, nt)
            nt = len(time_history)
            nx = len(x_flat)
            U_matrix = np.zeros((nx, nt))

            for i, U_t in enumerate(solution_history):
                U_matrix[:, i] = U_t.flatten()

            # 准备导出数据
            export_data = {
                'time_history': np.array(time_history),
                'x_coords': x_flat,
                'solution_history': U_matrix,
                'config_info': {
                    'time_scheme': getattr(self.config, 'time_scheme', 'unknown'),
                    'dt': self.config.dt,
                    'T': self.config.T,
                    'n_segments': self.config.n_segments,
                    'poly_degree': self.config.poly_degree,
                    'hidden_dims': self.config.hidden_dims
                }
            }

            # 导出到reference_data目录
            case_dir = getattr(self.config, 'case_dir', os.getcwd())
            ref_data_dir = os.path.join(case_dir, 'reference_data')
            os.makedirs(ref_data_dir, exist_ok=True)

            matlab_data_path = os.path.join(ref_data_dir, 'train_data.mat')
            scipy.io.savemat(matlab_data_path, export_data)

            print(f"  Train data exported to: {matlab_data_path}")
            print(f"  Data dimensions: {nt} time steps, {nx} spatial points")
            print(f"  Time range: [{time_history[0]:.6f}, {time_history[-1]:.6f}]")
            print(f"  Spatial range: [{x_flat.min():.6f}, {x_flat.max():.6f}]")

        except ImportError:
            print("  Warning: scipy.io not available, cannot export MATLAB data")
        except Exception as e:
            print(f"  Warning: Failed to export MATLAB data: {e}")

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
