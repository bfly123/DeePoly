import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from src.abstract_class.config.base_visualize import BaseVisualizer
import torch
from PIL import Image
import io
import os
from matplotlib import animation

class TimePDEVisualizer(BaseVisualizer):
    """Visualizer for time-dependent PDE problems - supports independent 1D and 2D visualization"""
    
    def __init__(self, config):
        super().__init__(config)
        # Unified dimension handling - eliminate dimension-based branching
        self.n_dim = len(config.spatial_vars) if hasattr(config, 'spatial_vars') else len(config.x_domain)

        plt.ion()

        # Unified figure initialization regardless of dimension
        figure_config = {
            1: {"figsize": (6, 4), "projection": None},
            2: {"figsize": (6, 6), "projection": "3d"}
        }

        config_2d = figure_config.get(self.n_dim, figure_config[1])
        self.fig = plt.figure(figsize=config_2d["figsize"])

        subplot_args = [111]
        if config_2d["projection"]:
            subplot_args.append({"projection": config_2d["projection"]})
            self.ax = self.fig.add_subplot(*subplot_args)
        else:
            self.ax = self.fig.add_subplot(*subplot_args)

        # Setup紧凑layout
        self.fig.tight_layout(pad=1.0)

        self._frames = []  # 用于Storeanimation帧
        self._save_gif = False
        self._visualization_interval = 1
        self._iteration_counter = 0
        self.cbar = None

        # 用于tracking scatter plot和line plotobject（Real-timeanimation）
        self._scatter_numerical = None
        self._line_reference = None
        self._step_counter = 0
        
    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """Unified solution plotting - eliminate dimension-based branching"""
        # Unified plotting dispatch
        plot_handlers = {
            1: self._plot_solution_1d,
            2: self._plot_solution_2d
        }

        handler = plot_handlers.get(self.n_dim, plot_handlers[1])
        handler(data, prediction, save_path)
    
    def _plot_solution_1d(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """1DSolution的Visualization"""
        fig = self._create_figure(figsize=(14, 5))
        
        # Get全局Solution
        x_global = np.vstack(data['x_segments'])
        u_global = np.vstack(data['u_segments'])
        pred_global = prediction
        
        # Sort以便Correct绘制LineGraph
        x = x_global[:, 0]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        u_sorted = u_global.flatten()[sort_idx]
        pred_sorted = pred_global.flatten()[sort_idx]
        
        # Create三个子Graph：True实Solution、PredictionSolution、误差
        ax1 = fig.add_subplot(131)
        ax1.plot(x_sorted, u_sorted, 'b-', linewidth=2, label='True实Solution')
        ax1.set_title('True实Solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2 = fig.add_subplot(132)
        ax2.plot(x_sorted, pred_sorted, 'r-', linewidth=2, label='PredictionSolution')
        ax2.set_title('PredictionSolution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 对比Graph
        ax3 = fig.add_subplot(133)
        ax3.plot(x_sorted, u_sorted, 'b-', linewidth=2, label='True实Solution', alpha=0.7)
        ax3.plot(x_sorted, pred_sorted, 'r--', linewidth=2, label='PredictionSolution', alpha=0.7)
        ax3.set_title('对比')
        ax3.set_xlabel('x')
        ax3.set_ylabel('u(x)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 添加段BoundaryMark
        if hasattr(self.config, 'n_segments') and hasattr(self.config, 'x_domain'):
            x_min, x_max = self.config.x_domain[0]
            n_seg = self.config.n_segments[0] if isinstance(self.config.n_segments, (list, tuple)) else self.config.n_segments
            segment_boundaries = np.linspace(x_min, x_max, n_seg + 1)
            
            for ax in [ax1, ax2, ax3]:
                for boundary in segment_boundaries[1:-1]:  # 跳过首TailBoundary
                    ax.axvline(boundary, color='k', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
    
    def _plot_solution_2d(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """2DSolution的Visualization"""
        fig = self._create_figure(figsize=(12, 5))
        
        # Get全局Solution
        x_global = np.vstack(data['x_segments'])
        u_global = np.vstack(data['u_segments'])
        pred_global = prediction
        
        # CreateMesh
        x = np.unique(x_global[:, 0])
        y = np.unique(x_global[:, 1])
        X, Y = np.meshgrid(x, y)
        
        # 重塑PredictionResult
        pred_grid = pred_global.reshape(len(y), len(x))
        u_grid = u_global.reshape(len(y), len(x))
        
        # True实Solution
        ax1 = fig.add_subplot(121)
        im1 = ax1.pcolormesh(X, Y, u_grid, cmap='viridis')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('True实Solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        
        # PredictionSolution
        ax2 = fig.add_subplot(122)
        im2 = ax2.pcolormesh(X, Y, pred_grid, cmap='viridis')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('PredictionSolution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        
        # 添加段Boundary
        self._add_segment_boundaries_2d([ax1, ax2])
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
        
    def plot_error(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """绘制误差分布 - According toDimensions自动选择1D或2DVisualization"""
        if self.n_dim == 1:
            self._plot_error_1d(data, prediction, save_path)
        else:
            self._plot_error_2d(data, prediction, save_path)
    
    def _plot_error_1d(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """1D误差分布Visualization"""
        fig = self._create_figure(figsize=(12, 8))
        
        # Compute误差
        x_global = np.vstack(data['x_segments'])
        u_global = np.vstack(data['u_segments'])
        error = np.abs(prediction - u_global)
        relative_error = error / (np.abs(u_global) + 1e-12)  # 避免除零
        
        # Sort
        x = x_global[:, 0]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        error_sorted = error.flatten()[sort_idx]
        rel_error_sorted = relative_error.flatten()[sort_idx]
        u_sorted = u_global.flatten()[sort_idx]
        pred_sorted = prediction.flatten()[sort_idx]
        
        # Create四个子Graph
        ax1 = fig.add_subplot(221)
        ax1.plot(x_sorted, error_sorted, 'r-', linewidth=2, label='Absolute Error')
        ax1.set_title('Absolute Error Distribution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('|u_true - u_pred|')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        ax2 = fig.add_subplot(222)
        ax2.plot(x_sorted, rel_error_sorted, 'g-', linewidth=2, label='Relative Error')
        ax2.set_title('Relative Error Distribution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('|u_true - u_pred|/|u_true|')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 误差Statistics
        ax3 = fig.add_subplot(223)
        ax3.semilogy(x_sorted, np.abs(u_sorted), 'b-', linewidth=2, label='|True实Solution|', alpha=0.7)
        ax3.semilogy(x_sorted, np.abs(pred_sorted), 'r--', linewidth=2, label='|PredictionSolution|', alpha=0.7)
        ax3.set_title('Solution的量级对比')
        ax3.set_xlabel('x')
        ax3.set_ylabel('|u|')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 误差直方Graph
        ax4 = fig.add_subplot(224)
        ax4.hist(np.log10(error_sorted + 1e-16), bins=30, alpha=0.7, edgecolor='black')
        ax4.set_title('Error Distribution Histogram')
        ax4.set_xlabel('log10(Absolute Error)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 添加段BoundaryMark
        if hasattr(self.config, 'n_segments') and hasattr(self.config, 'x_domain'):
            x_min, x_max = self.config.x_domain[0]
            n_seg = self.config.n_segments[0] if isinstance(self.config.n_segments, (list, tuple)) else self.config.n_segments
            segment_boundaries = np.linspace(x_min, x_max, n_seg + 1)
            
            for ax in [ax1, ax2, ax3]:
                for boundary in segment_boundaries[1:-1]:  # 跳过首TailBoundary
                    ax.axvline(boundary, color='k', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
    
    def _plot_error_2d(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        """2D误差分布Visualization"""
        fig = self._create_figure(figsize=(15, 5))
        
        # Compute误差
        x_global = np.vstack(data['x_segments'])
        u_global = np.vstack(data['u_segments'])
        error = np.abs(prediction - u_global)
        relative_error = error / (np.abs(u_global) + 1e-12)
        
        # CreateMesh
        x = np.unique(x_global[:, 0])
        y = np.unique(x_global[:, 1])
        X, Y = np.meshgrid(x, y)
        
        # 重塑误差
        error_grid = error.reshape(len(y), len(x))
        rel_error_grid = relative_error.reshape(len(y), len(x))
        
        # 绝对误差
        ax1 = fig.add_subplot(131)
        im1 = ax1.pcolormesh(X, Y, error_grid, cmap='hot')
        plt.colorbar(im1, ax=ax1, label='Absolute Error')
        ax1.set_title('Absolute Error Distribution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        
        # Phase对误差
        ax2 = fig.add_subplot(132)
        im2 = ax2.pcolormesh(X, Y, rel_error_grid, cmap='plasma')
        plt.colorbar(im2, ax=ax2, label='Relative Error')
        ax2.set_title('Relative Error Distribution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        
        # 对数误差
        ax3 = fig.add_subplot(133)
        log_error_grid = np.log10(error_grid + 1e-16)
        im3 = ax3.pcolormesh(X, Y, log_error_grid, cmap='viridis')
        plt.colorbar(im3, ax=ax3, label='log10(Absolute Error)')
        ax3.set_title('Log Error Distribution')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        
        # 添加段Boundary
        self._add_segment_boundaries_2d([ax1, ax2, ax3])
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
        
    def plot_evolution_step(self, T: float, u: np.ndarray, x: np.ndarray, save_path: Optional[str] = None) -> None:
        """绘制Time演化Process中的SingleTime step - According toDimensions自动选择Visualization方式"""
        if self.n_dim == 1:
            self._plot_evolution_step_1d(T, u, x, save_path)
        else:
            self._plot_evolution_step_2d(T, u, x, save_path)
    
    def _plot_evolution_step_1d(self, T: float, u: np.ndarray, x: np.ndarray, save_path: Optional[str] = None) -> None:
        """1DTime演化StepVisualization"""
        self.ax.clear()
        
        # Sort以便Correct绘制LineGraph
        sort_idx = np.argsort(x[:, 0])
        x_sorted = x[sort_idx, 0]
        u_sorted = u[sort_idx, 0] if u.ndim > 1 else u[sort_idx]
        
        # 绘制SolutionCurve
        self.ax.plot(x_sorted, u_sorted, 'b-', linewidth=3, label=f'u(x, t={T:.3f})')
        self.ax.set_xlabel("x", fontsize=12)
        self.ax.set_ylabel("u", fontsize=12)
        self.ax.set_title(f"Solution at T = {T:.4f}", fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # SetupcoordinateAxisRange
        if hasattr(self.config, 'x_domain'):
            self.ax.set_xlim([self.config.x_domain[0][0], self.config.x_domain[0][1]])
        
        # 动态AdjustmentyAxisRange
        u_min, u_max = u_sorted.min(), u_sorted.max()
        margin = (u_max - u_min) * 0.1 if u_max != u_min else 0.1
        self.ax.set_ylim([u_min - margin, u_max + margin])
        
        # 添加段BoundaryMark
        if hasattr(self.config, 'n_segments') and hasattr(self.config, 'x_domain'):
            x_min_domain, x_max_domain = self.config.x_domain[0]
            n_seg = self.config.n_segments[0] if isinstance(self.config.n_segments, (list, tuple)) else self.config.n_segments
            segment_boundaries = np.linspace(x_min_domain, x_max_domain, n_seg + 1)
            
            for boundary in segment_boundaries[1:-1]:  # 跳过首TailBoundary
                self.ax.axvline(boundary, color='red', linestyle=':', alpha=0.6, linewidth=1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.draw()
        plt.pause(0.01)
    
    def _plot_evolution_step_2d(self, T: float, u: np.ndarray, x: np.ndarray, save_path: Optional[str] = None) -> None:
        """2DTime演化StepVisualization"""
        self.ax.clear()
        
        # 绘制3D散pointGraph
        u_vals = u[:, 0] if u.ndim > 1 else u
        scatter = self.ax.scatter(x[:, 0], x[:, 1], u_vals, c=u_vals, cmap="RdBu", s=15, alpha=0.8)
        
        # Setup视Corner和label
        self.ax.view_init(elev=30, azim=45)
        self.ax.set_xlabel("x", fontsize=12)
        self.ax.set_ylabel("y", fontsize=12)
        self.ax.set_zlabel("u", fontsize=12)
        self.ax.set_title(f"Solution at T = {T:.4f}", fontsize=14, fontweight='bold')
        
        # SetupcoordinateAxisRange
        if hasattr(self.config, 'x_domain'):
            self.ax.set_xlim([self.config.x_domain[0][0], self.config.x_domain[0][1]])
            self.ax.set_ylim([self.config.x_domain[1][0], self.config.x_domain[1][1]])
            
        # 动态AdjustmentzAxisRange
        zmin, zmax = u_vals.min(), u_vals.max()
        margin = (zmax - zmin) * 0.1 if zmax != zmin else 0.1
        self.ax.set_zlim([zmin - margin, zmax + margin])
        
        # 添加color条
        if not hasattr(self, '_colorbar_2d') or self._colorbar_2d is None:
            self._colorbar_2d = plt.colorbar(scatter, ax=self.ax, shrink=0.5, aspect=20)
            self._colorbar_2d.set_label('u', fontsize=12)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.draw()
        plt.pause(0.01)
    
    def plot_evolution_step_with_reference(self, T: float, u: np.ndarray, x: np.ndarray,
                                         u_ref: Optional[np.ndarray] = None, solver=None, save_path: Optional[str] = None) -> None:
        """Unified evolution step plotting - eliminate dimension branching"""
        evolution_handlers = {
            1: lambda: self._plot_evolution_step_with_reference_1d(T, u, x, u_ref, solver, save_path),
            2: lambda: self._plot_evolution_step_2d(T, u, x, save_path)
        }

        handler = evolution_handlers.get(self.n_dim, evolution_handlers[1])
        handler()
    
    def _plot_evolution_step_with_reference_1d(self, T: float, u: np.ndarray, x: np.ndarray,
                                             u_ref: Optional[np.ndarray] = None, solver=None, save_path: Optional[str] = None) -> None:
        """1DTime演化Step与Reference solution对比Visualization - Usingscatter plot方式（与standalone solver一致）"""

        # IfNot yetInitializescatterobject，先Initialize
        if self._scatter_numerical is None:
            self.ax.clear()

            # Adjustment子Graphposition以留Exit更多Space给title
            self.fig.subplots_adjust(top=0.88)

            # Getxcoordinate
            x_flat = x[:, 0] if x.ndim > 1 else x
            u_flat = u[:, 0] if u.ndim > 1 else u

            # Createscatter plot用于Numerical solution（红色）
            self._scatter_numerical = self.ax.scatter(x_flat, u_flat, c='r', label='Numerical', s=20)

            # Createline plot用于Reference solution（蓝色）
            if u_ref is not None and solver is not None and hasattr(solver, 'reference_solution') and solver.reference_solution is not None:
                # GetReference solution的True实xcoordinate
                x_ref_true = solver.reference_solution['x_ref']
                # UsingTrue实的Reference solutionxcoordinateEnter行绘制
                sort_idx = np.argsort(x_ref_true)
                x_ref_sorted = x_ref_true[sort_idx]
                u_ref_sorted = u_ref[sort_idx]

                self._line_reference, = self.ax.plot(x_ref_sorted, u_ref_sorted, 'b--',
                                                   label='Reference Solution', linewidth=2, alpha=0.7)
                print(f"    Reference initialized with {len(x_ref_true)} points, x_range=[{x_ref_true.min():.3f}, {x_ref_true.max():.3f}]")
            else:
                self._line_reference = None
                print("    No reference solution available for initialization")

            # SetupGraph形attribute（动态Adjustment域Boundary）
            # FromData中Getx和u的Range
            x_min, x_max = x_flat.min(), x_flat.max()
            u_min, u_max = u_flat.min(), u_flat.max()

            # If有Reference solution，考虑Reference solution的Range
            if u_ref is not None and solver is not None and hasattr(solver, 'reference_solution') and solver.reference_solution is not None:
                x_ref_true = solver.reference_solution['x_ref']
                u_min = min(u_min, u_ref.min())
                u_max = max(u_max, u_ref.max())
                x_min = min(x_min, x_ref_true.min())
                x_max = max(x_max, x_ref_true.max())

            # 添加margin
            x_margin = (x_max - x_min) * 0.05
            u_margin = (u_max - u_min) * 0.1

            self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
            self.ax.set_ylim(u_min - u_margin, u_max + u_margin)
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('u')
            # 初Beginningtitle，Later会Update
            self.ax.set_title('Time PDE Real-time Solution', pad=20)
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)

        # Updatescatter plotData（Similarstandalone solver的line_pred.set_offsets）
        x_flat = x[:, 0] if x.ndim > 1 else x
        u_flat = u[:, 0] if u.ndim > 1 else u

        # UpdateNumerical solutionscatter plot的position
        self._scatter_numerical.set_offsets(np.column_stack((x_flat, u_flat)))

        # 动态AdjustmentyAxisRange（考虑CurrentNumerical solution的Range）
        u_min_current, u_max_current = u_flat.min(), u_flat.max()

        # UpdateReference solution（If有NewReference solutionData）
        if u_ref is not None and hasattr(self, '_line_reference') and self._line_reference is not None:
            if solver is not None and hasattr(solver, 'reference_solution') and solver.reference_solution is not None:
                # UsingTrue实的Reference solutionxcoordinate
                x_ref_true = solver.reference_solution['x_ref']
                sort_idx = np.argsort(x_ref_true)
                x_ref_sorted = x_ref_true[sort_idx]
                u_ref_sorted = u_ref[sort_idx]

                self._line_reference.set_data(x_ref_sorted, u_ref_sorted)

                # Meanwhile考虑Reference solution的RangeCome动态AdjustmentyAxis
                u_min_current = min(u_min_current, u_ref_sorted.min())
                u_max_current = max(u_max_current, u_ref_sorted.max())

                # Compute误差：Need将Reference solutionInterpolationToNumerical solutionMeshUp
                from scipy.interpolate import interp1d
                interp_func = interp1d(x_ref_true, u_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
                u_ref_interp = interp_func(x_flat)
                error = np.abs(u_flat - u_ref_interp)
                max_error = np.max(error)
                l2_error = np.sqrt(np.mean(error**2))

                # Increase步数计数器（AtdisplayBeforeIncrement以displayCorrect的步数）
                if not hasattr(self, "_step_counter"):
                    self._step_counter = 0
                if not hasattr(self, "_initial_time"):
                    self._initial_time = T
                if not hasattr(self, "_last_time"):
                    self._last_time = T

                # ComputeTime step size dt
                dt = T - self._last_time if self._step_counter > 0 else 0
                self._last_time = T
                self._step_counter += 1

                # UpdatetitleIncludeTime、Time step和误差information
                title_lines = [
                    f'Time: T = {T:.4f}s, Step: {self._step_counter}, dt = {dt:.4f}s',
                    f'Max Error: {max_error:.2e}, L2 Error: {l2_error:.2e}'
                ]
                self.ax.set_title('\n'.join(title_lines), pad=15, fontsize=11)

                # 添加详细的误差诊断information（仅AtForward几步）
                if self._step_counter <= 5:
                    print(f"  详细误差诊断 - Step {self._step_counter}, T={T:.3f}:")
                    print(f"    Numerical solutionRange: [{np.min(u_flat):.6f}, {np.max(u_flat):.6f}]")
                    print(f"    Reference solutionRange: [{np.min(u_ref_interp):.6f}, {np.max(u_ref_interp):.6f}]")
                    print(f"    Time同步Check: Numerical solutionT={T:.6f}, Reference solutionT={T:.6f} (应该Same)")
                    print(f"    最大误差: {max_error:.2e}, L2误差: {l2_error:.2e}")
                    print(f"    误差分布: Average={np.mean(error):.2e}, Standard deviation={np.std(error):.2e}")
            else:
                # 没有solver或Reference solution时
                if not hasattr(self, "_step_counter"):
                    self._step_counter = 0
                if not hasattr(self, "_initial_time"):
                    self._initial_time = T
                if not hasattr(self, "_last_time"):
                    self._last_time = T

                dt = T - self._last_time if self._step_counter > 0 else 0
                self._last_time = T
                self._step_counter += 1
                self.ax.set_title(f'Time: T = {T:.4f}s, Step: {self._step_counter}, dt = {dt:.4f}s', pad=15, fontsize=11)
        else:
            # 没有Reference solution时的title
            if not hasattr(self, "_step_counter"):
                self._step_counter = 0
            if not hasattr(self, "_initial_time"):
                self._initial_time = T
            if not hasattr(self, "_last_time"):
                self._last_time = T

            dt = T - self._last_time if self._step_counter > 0 else 0
            self._last_time = T
            self._step_counter += 1
            self.ax.set_title(f'Time: T = {T:.4f}s, Step: {self._step_counter}, dt = {dt:.4f}s', pad=15, fontsize=11)

        # 动态AdjustmentyAxisRange（AtAllDataUpdateCompleteBackward）
        if u_max_current > u_min_current:  # 避免除零Error
            u_margin = (u_max_current - u_min_current) * 0.1
            self.ax.set_ylim(u_min_current - u_margin, u_max_current + u_margin)
        else:
            # IfRange为零，UsingDefault的小Range
            self.ax.set_ylim(u_min_current - 0.1, u_max_current + 0.1)

        # 刷新display（Similarstandalone solver的方式）
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # 小的延时以便平滑animation（与standalone solver一致）
        import time
        time.sleep(0.05)

        # SaveGraph像（IfNeed）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    def close_evolution_plot(self) -> None:
        """Shutdown time演化绘Graph"""
        plt.ioff()
        if hasattr(self, 'fig'):
            plt.close(self.fig)

    def finalize_realtime_animation(self) -> None:
        """CompleteReal-timeanimationdisplay"""
        plt.ioff()
        print("Time evolution completed. Close the plot window to continue...")
        
    def plot_mesh_results(
        self,
        data: Dict,
        model: torch.nn.Module,
        preds: np.ndarray,
        title: Optional[str] = None,
        save_gif: bool = False,
        save_png: bool = False,
        gif_filename: str = "animation.gif",
        png_filename: Optional[str] = None,
        fast_mode: bool = False,
        contour_levels: int = 50,
    ) -> None:
        """Visualizationu分量的流场分布"""
        self._save_gif = save_gif
        
        # 提取ConfigurationParameter
        ns1, ns2 = self.config.n_segments
        np1, np2 = self.config.points_domain_test
        
        # CreateMesh
        x1 = np.linspace(self.config.x_domain[0, 0], self.config.x_domain[0, 1], np1 * ns1)
        x2 = np.linspace(self.config.x_domain[1, 0], self.config.x_domain[1, 1], np2 * ns2)
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        
        # 重GroupPredictionvalueToMeshShape
        u_pred = preds[:, 0].reshape(np2 * ns2, np1 * ns1)
        
        # Create或UpdateGraph表
        if not hasattr(self, '_mesh_fig'):
            self._mesh_fig = plt.figure(figsize=(10, 8))
            self._mesh_axes = self._mesh_fig.add_subplot(111)
        
        # ClearCurrentGraph表
        self._mesh_axes.clear()
        
        # 绘制等valueLineGraph
        contf = self._mesh_axes.contourf(
            x1_grid, 
            x2_grid, 
            u_pred,
            levels=contour_levels,
            cmap='RdBu',
        )
        
        # Setuptitle和label
        if title:
            self._mesh_axes.set_title(title)
        self._mesh_axes.set_xlabel('x')
        self._mesh_axes.set_ylabel('y')
        self._mesh_axes.set_aspect('equal')
        
        # Update或添加color条
        if hasattr(self._mesh_axes, 'colorbar'):
            self._mesh_axes.colorbar.remove()
        self._mesh_axes.colorbar = plt.colorbar(contf, ax=self._mesh_axes)
        self._mesh_axes.colorbar.set_label('u')
        
        # Adjustmentlayout
        plt.tight_layout()
        
        # UpdateGraph表
        self._mesh_fig.canvas.draw()
        
        # SaveCurrent帧为PNG（IfNeed）
        if save_png and png_filename:
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        
        # 捕获Current帧用于GIF
        if self._save_gif:
            buf = io.BytesIO()
            self._mesh_fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            self._frames.append(Image.open(buf))
        
        self._mesh_fig.canvas.flush_events()
        plt.pause(0.01)
        
    def plot_mesh_results_final(
        self,
        data: Dict,
        model: torch.nn.Module,
        preds: np.ndarray,
    ) -> None:
        """绘制FinalResult：u、v、p分量Graph和黑白流LineGraph"""
        # 提取ConfigurationParameter
        ne = self.config.n_eqs
        ns1, ns2 = self.config.n_segments
        np1, np2 = self.config.points_per_segment_test
        
        # Compute总point数和区段
        points_per_section = np1 * np2
        total_sections = ns1 * ns2
        
        # InitializeMeshArray
        x1_grid = np.zeros((np1 * ns1, np2 * ns2))
        x2_grid = np.zeros((np1 * ns1, np2 * ns2))
        
        # paddingMeshArray
        for section in range(total_sections):
            section_x = section // ns2
            section_y = section % ns2
            
            for point in range(points_per_section):
                point_x = point // np2
                point_y = point % np2
                
                # GetMeshpoint
                grid_points = data["x_segments"][section][point]
                x1_grid[point_y + section_y * np1, point_x + section_x * np2] = grid_points[0]
                x2_grid[point_y + section_y * np1, point_x + section_x * np2] = grid_points[1]
        
        # Get唯一coordinate以确保MeshRule性
        x1_unique = np.unique(x1_grid)
        x2_unique = np.unique(x2_grid)
        
        # InitializePredictionArray
        u_pred = np.zeros((np1 * ns1, np2 * ns2, ne))
        
        # paddingPredictionArray
        for section in range(total_sections):
            for point in range(points_per_section):
                for eq in range(ne):
                    section_x = section // ns2
                    section_y = section % ns2
                    point_x = point // np2
                    point_y = point % np2
                    u_pred[point_y + section_y * np1, point_x + section_x * np2, eq] = (
                        preds[points_per_section * section + point, eq]
                    )
        
        # CreateResultDirectory
        os.makedirs('results', exist_ok=True)
        
        # 1. U分量Graph
        fig = self._create_figure(figsize=(8, 6))
        contf_u = plt.contourf(x1_grid, x2_grid, u_pred[:, :, 0], levels=20, cmap='viridis')
        plt.colorbar(contf_u, label='U Velocity')
        plt.title('U Component')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('results/u_component.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. V分量Graph
        fig = self._create_figure(figsize=(8, 6))
        contf_v = plt.contourf(x1_grid, x2_grid, u_pred[:, :, 1], levels=20, cmap='viridis')
        plt.colorbar(contf_v, label='V Velocity')
        plt.title('V Component')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('results/v_component.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. P分量Graph
        fig = self._create_figure(figsize=(8, 6))
        if ne >= 3:
            contf_p = plt.contourf(x1_grid, x2_grid, u_pred[:, :, 2], levels=20, cmap='viridis')
            plt.colorbar(contf_p, label='Pressure')
            plt.title('P Component')
        else:
            plt.title('Missing P Component Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('results/p_component.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 黑白流LineGraph
        fig = self._create_figure(figsize=(8, 6))
        u_component = u_pred[:, :, 0].T
        v_component = u_pred[:, :, 1].T
        
        plt.streamplot(
            x1_unique, 
            x2_unique, 
            u_component, 
            v_component, 
            color='k',
            linewidth=1.0,
            density=3.0,
            arrowsize=0,
        )
        plt.title('Streamlines')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('results/streamlines.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # OutputTecplotformatFile
        with open('results/flow_field.dat', 'w') as f:
            f.write('TITLE = "Flow Field Data"\n')
            f.write('VARIABLES = "X", "Y", "U", "V"')
            if ne >= 3:
                f.write(', "P"')
            f.write('\n')
            
            f.write(f'ZONE I={np1 * ns1}, J={np2 * ns2}, F=POINT\n')
            
            for j in range(np2 * ns2):
                for i in range(np1 * ns1):
                    line = f"{x1_grid[i,j]:.6f} {x2_grid[i,j]:.6f} {u_pred[i,j,0]:.6f} {u_pred[i,j,1]:.6f}"
                    if ne >= 3:
                        line += f" {u_pred[i,j,2]:.6f}"
                    f.write(line + '\n')
        
        print("4 separate images and Tecplot format data file have been saved to the results folder")
        
    def update_animation_data(self, it: int, T: float, data_train: Dict, model: torch.nn.Module,
                            coeffs: np.ndarray, fitter, solver=None, U_direct: np.ndarray = None, U_seg_direct: list = None) -> None:
        """UpdateanimationData - FromSolve器MoveToVisualization器，SupportReference solution对比，UsingTrainingData绘制scatter plot"""
        animation_skip = getattr(self.config, "animation_skip", 10)
        
        # CheckYesNoNeedUpdateanimationData
        if it % animation_skip == 0 or T >= getattr(self.config, 'T', float('inf')):
            # Update timeHistory
            if not hasattr(self, 'time_history_viz'):
                self.time_history_viz = []
                self.solution_history_viz = []
                
            self.time_history_viz.append(T)
            
            # GetCurrent time步的Solution - 直接UsingTime stepOutput的Solution
            try:
                # Check if data_train and required fields are available
                if data_train is None:
                    print(f"  Warning: data_train is None, skipping animation update")
                    return

                if 'x' not in data_train and 'x_segments' not in data_train:
                    print(f"  Warning: No coordinate data in data_train, skipping animation update")
                    return

                # UsingTime step直接Output的Solution，而不YesPass construct Re-Construct
                if U_direct is not None:
                    U_train = U_direct.copy()
                else:
                    # 备用Plan：If没有直接TransferSolution，则Using construct
                    if coeffs is None:
                        print(f"  Warning: coeffs is None at T={T:.4f}, skipping animation update")
                        return
                    U_train, _ = fitter.construct(data_train, model, coeffs)
                    print(f"  备用Plan：Using construct ConstructSolution")

                if U_train is None:
                    print(f"  Warning: U_train is None, skipping animation update")
                    return

                self.solution_history_viz.append(U_train.copy())
                
                # Optional择Real-time绘制Time演化Step
                if getattr(self.config, 'realtime_visualization', False):
                    print(f"  Real-time visualization enabled, plotting at T = {T:.4f}")
                    x_coords = data_train.get('x', data_train.get('x_segments', []))
                    if hasattr(x_coords, '__iter__') and len(x_coords) > 0:
                        if isinstance(x_coords, list):
                            x_plot = np.vstack(x_coords) if x_coords else np.array([[0]])
                        else:
                            x_plot = x_coords
                        
                        # GetReference solution（If有）
                        u_ref = None
                        if solver and hasattr(solver, 'get_reference_solution_at_time'):
                            u_ref = solver.get_reference_solution_at_time(T)
                            print(f"    Reference solution at T={T:.4f}: {'Yes' if u_ref is not None else 'No'}")
                            if u_ref is not None:
                                print(f"    Reference solution range: [{np.min(u_ref):.4f}, {np.max(u_ref):.4f}]")
                            
                        self.plot_evolution_step_with_reference(T, U_train, x_plot, u_ref, solver)
                        print(f"    Real-time plot updated for T = {T:.4f}")
                else:
                    print(f"  Real-time visualization disabled (setting: {getattr(self.config, 'realtime_visualization', 'Not found')})")
                        
                print(f"  animationData已Update: T = {T:.6f}, 总帧数 = {len(self.solution_history_viz)}")
                
            except Exception as e:
                print(f"  Warning: animationDataUpdateFail: {e}")
    
    def get_animation_data(self):
        """GetanimationData"""
        if hasattr(self, 'time_history_viz') and hasattr(self, 'solution_history_viz'):
            return self.time_history_viz, self.solution_history_viz
        else:
            return [], []
    
    def save_animation(self, filename: str = "animation.gif", duration: int = 200) -> None:
        """将捕获的帧Save为GIFanimation"""
        if len(self._frames) > 0:
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
            
            self._frames[0].save(
                filename,
                save_all=True,
                append_images=self._frames[1:],
                optimize=True,
                duration=duration,
                loop=0,
            )
            print(f"GIFanimation已Save至 {filename}")
            
            self._frames = []
        else:
            print("没有帧可Save")
    
    def create_time_evolution_gif(self, time_history: List[float], solution_history: List[np.ndarray],
                                data_train: Dict, filename: str = "time_evolution.gif", solver=None) -> None:
        """CreateTime演化GIFanimation，SupportReference solution对比，UsingTrainingDatacoordinate"""
        if not time_history or not solution_history:
            print("Warning: 没有Time演化Data可Createanimation")
            return
            
        try:
            print(f"正AtCreateTime演化animation: {filename}")
            
            # Getxcoordinate
            if 'x_segments' in data_train:
                x_coords = np.vstack(data_train['x_segments'])
            elif 'x' in data_train:
                x_coords = data_train['x']
            else:
                print("Warning: 无法GetxcoordinateData")
                return
            
            # 清空Before的帧
            self._frames = []
            
            # 为EachTime stepGenerate帧
            for i, (T, solution) in enumerate(zip(time_history, solution_history)):
                # Unified figure creation - eliminate dimension branching
                figure_sizes = {1: (10, 6), 2: (10, 8)}
                figsize = figure_sizes.get(self.n_dim, figure_sizes[1])
                temp_fig = plt.figure(figsize=figsize)
                temp_fig.subplots_adjust(top=0.88)  # 留ExitSpace给title
                
                if self.n_dim == 1:
                    # 1DStatus - UsingscatterMaintain与Real-timeanimation一致
                    x_flat = x_coords[:, 0] if x_coords.ndim > 1 else x_coords
                    u_flat = solution.flatten()

                    ax = temp_fig.add_subplot(111)
                    # UsingscatterdisplayNumerical solution（红色point，与Real-timeanimation一致）
                    ax.scatter(x_flat, u_flat, c='r', label='Numerical', s=20)
                    
                    # 添加Reference solution对比（If有solver且有Reference solution）
                    if solver and hasattr(solver, 'get_reference_solution_at_time'):
                        u_ref = solver.get_reference_solution_at_time(T)
                        if u_ref is not None:
                            # UsingReference solution的原Beginningxcoordinate
                            if hasattr(solver, 'reference_solution') and solver.reference_solution is not None:
                                try:
                                    x_ref = solver.reference_solution['x_ref']
                                    # 对Reference solutionSort以画Exit平滑的Line
                                    sort_idx = np.argsort(x_ref)
                                    x_ref_sorted = x_ref[sort_idx]
                                    u_ref_sorted = u_ref[sort_idx]

                                    # UsingLine条displayReference solution（蓝色虚Line，与Real-timeanimation一致）
                                    ax.plot(x_ref_sorted, u_ref_sorted, 'b--', linewidth=2, alpha=0.7, label='Reference Solution')

                                    # Compute误差（NeedInterpolationReference solutionToNumerical solutionMesh）
                                    from scipy.interpolate import interp1d
                                    interp_func = interp1d(x_ref, u_ref, kind='cubic',
                                                          bounds_error=False, fill_value='extrapolate')
                                    u_ref_interp = interp_func(x_flat)

                                    error = np.abs(u_flat - u_ref_interp)
                                    max_error = np.max(error)
                                    l2_error = np.sqrt(np.mean(error**2))
                                except Exception as e:
                                    print(f"Reference solutionInterpolationFail: {e}")
                    
                    ax.set_xlabel('x', fontsize=12)
                    ax.set_ylabel('u', fontsize=12)

                    # ComputeTime step size
                    if i > 0:
                        dt = time_history[i] - time_history[i-1]
                    else:
                        dt = time_history[1] - time_history[0] if len(time_history) > 1 else 0

                    # UpdatetitleIncludeTime、Time step和步数information
                    title_text = f'Time: T = {T:.4f}s, Step: {i+1}, dt = {dt:.4f}s'

                    # If有误差information，将其MoveTotitle
                    if 'error' in locals() and 'max_error' in locals() and 'l2_error' in locals():
                        title_text += f'\nMax Error: {max_error:.2e}, L2 Error: {l2_error:.2e}'
                        # Remove单独的误差文本框（AlreadyAttitle中display）
                        # ax.text已被Remove

                    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=15)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right')
                    
                    # Setup固定的yAxisRange以便animationStable
                    all_solutions = np.concatenate([s.flatten() for s in solution_history])
                    y_min, y_max = all_solutions.min(), all_solutions.max()

                    # If有Reference solution，也考虑其Range
                    if solver and hasattr(solver, 'reference_solution') and solver.reference_solution is not None:
                        try:
                            u_ref_all = solver.reference_solution['u_ref']
                            y_min = min(y_min, u_ref_all.min())
                            y_max = max(y_max, u_ref_all.max())
                        except:
                            pass

                    # 添加margin
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim([y_min - margin, y_max + margin])

                    # SetupxAxisRange
                    x_min, x_max = x_flat.min(), x_flat.max()
                    if solver and hasattr(solver, 'reference_solution') and solver.reference_solution is not None:
                        try:
                            x_ref = solver.reference_solution['x_ref']
                            x_min = min(x_min, x_ref.min())
                            x_max = max(x_max, x_ref.max())
                        except:
                            pass
                    x_margin = (x_max - x_min) * 0.05
                    ax.set_xlim([x_min - x_margin, x_max + x_margin])
                    
                else:
                    # 2DStatus
                    ax = temp_fig.add_subplot(111, projection='3d')
                    u_vals = solution.flatten()
                    ax.scatter(x_coords[:, 0], x_coords[:, 1], u_vals, c=u_vals, cmap='RdBu', s=10)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y') 
                    ax.set_zlabel('u')
                    ax.set_title(f'T = {T:.4f}')
                
                plt.tight_layout()
                
                # 将Current帧添加Toanimation
                buf = io.BytesIO()
                temp_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                self._frames.append(Image.open(buf))
                plt.close(temp_fig)
                
                if (i + 1) % 10 == 0:  # 每10帧打印Progress
                    print(f"  已Generate {i + 1}/{len(time_history)} 帧")
            
            # SaveGIFanimation
            self.save_animation(filename, duration=200)
            print(f"Time演化animationCreateComplete: {filename}")
            
        except Exception as e:
            print(f"animationCreateFail: {e}")
            
    def _add_segment_boundaries_2d(self, axes_list: List) -> None:
        """为2DGraph添加段Boundary - Auxiliarymethod"""
        if hasattr(self.config, 'x_min') and hasattr(self.config, 'x_max'):
            for n in range(getattr(self, 'Ns', 0)):
                try:
                    x_min = self.config.x_min[n]
                    x_max = self.config.x_max[n]
                    
                    # 绘制段Boundary
                    for ax in axes_list:
                        ax.plot([x_min[0], x_max[0]], [x_min[1], x_min[1]], 'k--', alpha=0.4, linewidth=1)
                        ax.plot([x_min[0], x_max[0]], [x_max[1], x_max[1]], 'k--', alpha=0.4, linewidth=1)
                        ax.plot([x_min[0], x_min[0]], [x_min[1], x_max[1]], 'k--', alpha=0.4, linewidth=1)
                        ax.plot([x_max[0], x_max[0]], [x_min[1], x_max[1]], 'k--', alpha=0.4, linewidth=1)
                except (IndexError, AttributeError):
                    # Skip boundary drawing if segment boundaries not available
                    pass
    
    def plot_loss_history(self, loss_history: List[float], save_path: Optional[str] = None) -> None:
        """绘制TrainingLossHistory"""
        fig = self._create_figure(figsize=(10, 6))
        
        iterations = range(1, len(loss_history) + 1)
        ax = fig.add_subplot(111)
        ax.semilogy(iterations, loss_history, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Iterate步数', fontsize=12)
        ax.set_ylabel('Lossvalue (对数尺Degree)', fontsize=12)
        ax.set_title('TrainingLossHistory', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加Statisticsinformation
        if loss_history:
            final_loss = loss_history[-1]
            min_loss = min(loss_history)
            ax.text(0.02, 0.98, f'FinalLoss: {final_loss:.2e}\n最小Loss: {min_loss:.2e}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
    
    def plot_time_evolution_summary(self, time_history: List[float], solution_history: List[np.ndarray], 
                                  save_path: Optional[str] = None) -> None:
        """绘制Time演化SummaryGraph"""
        if not time_history or not solution_history:
            print("Warning: 没有Time演化Data可绘制")
            return
            
        if self.n_dim == 1:
            self._plot_time_evolution_summary_1d(time_history, solution_history, save_path)
        else:
            self._plot_time_evolution_summary_2d(time_history, solution_history, save_path)
    
    def _plot_time_evolution_summary_1d(self, time_history: List[float], solution_history: List[np.ndarray], 
                                       save_path: Optional[str] = None) -> None:
        """1DTime演化SummaryGraph"""
        fig = self._create_figure(figsize=(15, 10))
        
        # 确保DataValidity
        if len(time_history) != len(solution_history):
            print(f"Warning: TimeHistoryLength({len(time_history)})与SolutionHistoryLength({len(solution_history)})不匹配")
            min_len = min(len(time_history), len(solution_history))
            time_history = time_history[:min_len]
            solution_history = solution_history[:min_len]
        
        if len(time_history) == 0:
            print("Warning: 没有Time演化Data")
            return
            
        # 选择Several代表性Timepoint
        n_snapshots = min(6, len(time_history))
        if len(time_history) == 1:
            snapshot_indices = [0]
        else:
            snapshot_indices = np.linspace(0, len(time_history)-1, n_snapshots, dtype=int)
        
        # Create子GraphMesh
        for i, idx in enumerate(snapshot_indices):
            ax = fig.add_subplot(2, 3, i+1)
            u = solution_history[idx]
            t = time_history[idx]
            
            # 这InsideNeedxcoordinateinformation，AssumptionFromconfigGet
            if hasattr(self.config, 'x_domain') and hasattr(self.config, 'points_domain_test'):
                n_points = self.config.points_domain_test[0] if isinstance(self.config.points_domain_test, (list, tuple)) else self.config.points_domain_test
                x = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], len(u))
                ax.plot(x, u.flatten(), 'b-', linewidth=2)
            else:
                ax.plot(u.flatten(), 'b-', linewidth=2)
            
            ax.set_title(f'T = {t:.4f}', fontsize=12)
            ax.grid(True, alpha=0.3)
            if i >= 3:  # Down排子Graph
                ax.set_xlabel('x', fontsize=10)
            if i % 3 == 0:  # Left列子Graph
                ax.set_ylabel('u', fontsize=10)
        
        plt.suptitle('Time演化快照', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
    
    def _plot_time_evolution_summary_2d(self, time_history: List[float], solution_history: List[np.ndarray], 
                                       save_path: Optional[str] = None) -> None:
        """2DTime演化SummaryGraph"""
        fig = self._create_figure(figsize=(15, 10))
        
        # 确保DataValidity
        if len(time_history) != len(solution_history):
            print(f"Warning: TimeHistoryLength({len(time_history)})与SolutionHistoryLength({len(solution_history)})不匹配")
            min_len = min(len(time_history), len(solution_history))
            time_history = time_history[:min_len]
            solution_history = solution_history[:min_len]
        
        if len(time_history) == 0:
            print("Warning: 没有Time演化Data")
            return
        
        # 选择Several代表性Timepoint
        n_snapshots = min(6, len(time_history))
        if len(time_history) == 1:
            snapshot_indices = [0]
        else:
            snapshot_indices = np.linspace(0, len(time_history)-1, n_snapshots, dtype=int)
        
        for i, idx in enumerate(snapshot_indices):
            ax = fig.add_subplot(2, 3, i+1)
            u = solution_history[idx]
            t = time_history[idx]
            
            # 简单的2D热ForceGraphdisplay
            if u.ndim == 1:
                # AssumptionYesRuleMesh，Need重塑
                side_length = int(np.sqrt(len(u)))
                if side_length * side_length == len(u):
                    u_grid = u.reshape(side_length, side_length)
                    im = ax.imshow(u_grid, cmap='viridis', aspect='equal')
                    plt.colorbar(im, ax=ax, shrink=0.8)
            
            ax.set_title(f'T = {t:.4f}', fontsize=12)
            ax.set_aspect('equal')
        
        plt.suptitle('Time演化快照', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)
    
    def set_visualization_interval(self, interval: int = 1) -> None:
        """SetupVisualizationBetween隔"""
        self._visualization_interval = max(1, interval)
        print(f"VisualizationBetween隔Setup为: 每 {self._visualization_interval} 次IterateUpdateOnce")
    
    def reset_animation_data(self):
        """ResetanimationData"""
        self.time_history_viz = []
        self.solution_history_viz = []
        self._frames = []
        print("animationData已Reset")
        
    def examin_data(self, data_train: Dict) -> None:
        """CheckData分布"""
        fig = self._create_figure()
        plt.scatter(
            data_train["x_boundary"][:, 0],
            data_train["x_boundary"][:, 1],
            label="BoundarySolution",
        )
        plt.scatter(data_train["x"][:, 0], data_train["x"][:, 1], label="ExactSolution")
        plt.legend()
        plt.show()
        
    def examin_net(self, model: torch.nn.Module, data_GPU: torch.Tensor) -> None:
        """CheckNetworkOutput并绘制流场"""
        model.eval()
        with torch.no_grad():
            # CreateRuleMesh
            N = 100
            x1_grid = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], N)
            x2_grid = np.linspace(self.config.x_domain[1][0], self.config.x_domain[1][1], N)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            grid_points = np.column_stack((X1.ravel(), X2.ravel()))
            
            # Convert为Tensor
            grid_points = torch.tensor(
                grid_points, dtype=torch.float64, device=self.config.device, requires_grad=True
            )
            
            # GetNetworkPrediction
            _, u_pred = model(grid_points)
            
            # SeparateSpeed和Pressure分量
            u = u_pred[..., 0]
            
            # Convert为numpyArray
            u = u.detach().cpu().numpy()
            
            # Re-Adjustment为MeshShape
            U = u.reshape(N, N)
            
            # CreateGraph表
            fig = self._create_figure(figsize=(15, 5))
            
            # 绘制U分量
            ax1 = fig.add_subplot(111)
            cs1 = ax1.contourf(X1, X2, U, levels=50, cmap="RdBu")
            plt.colorbar(cs1, ax=ax1, label="USpeed")
            ax1.set_title("USpeed")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            
            plt.tight_layout()
            plt.show() 