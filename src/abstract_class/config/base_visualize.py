import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import os
import math
import torch
import matplotlib.gridspec as gridspec

# Set global font and style settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False
})


class BaseVisualizer:
    """Base visualization class"""

    def __init__(self, config):
        self.config = config
        self.n_dim = config.n_dim
        self.n_eqs = config.n_eqs
        self.Ns = np.prod(config.n_segments)

    def _create_figure(self, figsize: tuple = (10, 8)) -> plt.Figure:
        """Create figure"""
        # No longer use seaborn style, rely on global rcParams settings
        return plt.figure(figsize=figsize)

    def _save_figure(self, fig: plt.Figure, save_path: str) -> None:
        """Save figure"""
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", 
                       facecolor='white', edgecolor='none')

    def _close_figure(self, fig: plt.Figure) -> None:
        """Close figure"""
        plt.close(fig)

    def _get_segment_boundaries(self, data: Dict, segment_idx: int) -> Dict:
        """Get segment boundaries"""
        return {
            "x_min": data["x_min"][segment_idx],
            "x_max": data["x_max"][segment_idx],
        }

    def _normalize_to_physical(
        self, x_norm: np.ndarray, segment_idx: int
    ) -> np.ndarray:
        """Convert normalized coordinates back to physical coordinates"""
        x_min = self.config.x_min[segment_idx]
        x_max = self.config.x_max[segment_idx]
        return x_norm * (x_max - x_min) + x_min

    def calculate_errors(self, pred: np.ndarray, exact: np.ndarray) -> Dict[str, float]:
        """Calculate prediction error metrics"""
        mse = np.mean((pred - exact) ** 2)
        mae = np.mean(np.abs(pred - exact))
        max_error = np.max(np.abs(pred - exact))
        rel_error = np.mean(np.abs((pred - exact) / (exact + 1e-12)))
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae, 
            'Max Error': max_error,
            'Relative Error': rel_error,
            'RMSE': rmse
        }

    def get_model_predictions(self, model: torch.nn.Module, data: Dict) -> np.ndarray:
        """Get model prediction results"""
        x = data["x"]
        with torch.no_grad():
            _, predictions = model(torch.tensor(x, dtype=torch.float64))
        return predictions.detach().numpy()

    def get_deepoly_predictions(self, fitter, data: Dict, model: torch.nn.Module, coeffs: np.ndarray) -> np.ndarray:
        """Get DeePoly prediction results"""
        predictions, _ = fitter.construct(data, model, coeffs)
        return predictions

    def reconstruct_grid_nd(self, data: Dict, preds: np.ndarray, config) -> tuple:
        """Reconstruct regular grid from segmented data for arbitrary dimensions
        
        Args:
            data: Dictionary containing segmented data
            preds: Predictions array with shape (total_points, n_eqs)
            config: Configuration object
            
        Returns:
            tuple: (coord_grids, u_pred) where:
                - coord_grids: List of coordinate grids for each dimension
                - u_pred: Prediction array with shape (*grid_dims, n_eqs)
        """
        n_dim = config.n_dim
        n_eqs = config.n_eqs
        total_sections = np.prod(config.n_segments)
        
        # Get grid dimensions from config
        if hasattr(config, 'points_domain_test') and isinstance(config.points_domain_test, list):
            grid_dims = config.points_domain_test
        else:
            # Fallback: assume equal dimensions
            total_points = sum(len(data["x_segments"][i]) for i in range(total_sections))
            points_per_dim = int(total_points ** (1.0 / n_dim))
            grid_dims = [points_per_dim] * n_dim
        
        # Collect all coordinates and predictions
        all_coords = []
        all_preds = []
        
        pred_idx = 0
        for section in range(total_sections):
            n_points_section = len(data["x_segments"][section])
            
            section_coords = data["x_segments"][section]
            section_preds = preds[pred_idx:pred_idx + n_points_section, :]
            
            all_coords.append(section_coords)
            all_preds.append(section_preds)
            
            pred_idx += n_points_section
        
        # Combine all data
        all_coords = np.vstack(all_coords)
        all_preds = np.vstack(all_preds)
        
        # Find unique coordinates for each dimension
        unique_coords = []
        for dim in range(n_dim):
            coords_dim = all_coords[:, dim]
            unique_dim = np.sort(np.unique(coords_dim))
            unique_coords.append(unique_dim)
            
            # Verify grid dimensions
            if len(unique_dim) != grid_dims[dim]:
                print(f"Warning: Dimension {dim} - Expected {grid_dims[dim]} points, got {len(unique_dim)}")
                grid_dims[dim] = len(unique_dim)
        
        # Create coordinate grids
        coord_grids = np.meshgrid(*unique_coords, indexing='ij')
        
        # Initialize prediction grid
        u_pred = np.zeros(tuple(grid_dims) + (n_eqs,))
        
        # Create a mapping for efficient indexing
        coord_to_index = {}
        for i, coord in enumerate(all_coords):
            # Convert coordinates to indices
            indices = []
            for dim in range(n_dim):
                idx = np.searchsorted(unique_coords[dim], coord[dim])
                if idx >= grid_dims[dim]:
                    idx = grid_dims[dim] - 1
                indices.append(idx)
            
            # Store prediction at grid position
            u_pred[tuple(indices)] = all_preds[i]
        
        return coord_grids, u_pred

    def reconstruct_grid_test2D(self, data: Dict, preds: np.ndarray, config) -> tuple:
        """2D-specific wrapper for backward compatibility"""
        if config.n_dim != 2:
            raise ValueError("reconstruct_grid_test2D only supports 2D data")
        
        coord_grids, u_pred = self.reconstruct_grid_nd(data, preds, config)
        return coord_grids[0], coord_grids[1], u_pred

    def plot_2d_comparison(self, 
                          data_train: Dict,
                          data_test: Dict,
                          model: torch.nn.Module,
                          fitter,
                          coeffs: np.ndarray,
                          exact_solution_func: Callable,
                          result_dir: str,
                          variables: Optional[List[str]] = None,
                          timing_info: Optional[Dict[str, float]] = None) -> None:
        """
        Complete comparison plotting for 2D problems
        
        Args:
            data_train: Training data
            data_test: Test data  
            model: Trained model
            fitter: DeePoly fitter
            coeffs: DeePoly coefficients
            exact_solution_func: Exact solution function
            result_dir: Results save directory
            variables: List of variables to plot, defaults to all variables
            timing_info: Dictionary containing timing information (total_time, scoper_time, sniper_time)
        """
        # Default plot all variables
        if variables is None:
            if hasattr(self.config, 'vars_list'):
                variables = self.config.vars_list
            else:
                variables = [f'u_{i}' for i in range(self.n_eqs)]
        
        # Create visualization directory
        vis_dir = os.path.join(result_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get prediction results
        x_train = data_train["x"]
        x_test = data_test["x"]
        
        # Model predictions
        net_train = self.get_model_predictions(model, data_train)
        net_test = self.get_model_predictions(model, data_test)
        
        # Exact solutions
        exact_train = exact_solution_func(x_train)
        exact_test = exact_solution_func(x_test)
        
        # DeePoly predictions
        deePoly_train = self.get_deepoly_predictions(fitter, data_train, model, coeffs)
        deePoly_test = self.get_deepoly_predictions(fitter, data_test, model, coeffs)
        
        # Calculate errors
        net_train_errors = self.calculate_errors(net_train, exact_train)
        net_test_errors = self.calculate_errors(net_test, exact_test)
        deePoly_train_errors = self.calculate_errors(deePoly_train, exact_train)
        deePoly_test_errors = self.calculate_errors(deePoly_test, exact_test)
        
        # Print error statistics
        self._print_error_statistics(net_train_errors, net_test_errors, 
                                   deePoly_train_errors, deePoly_test_errors)
        
        # Save detailed error report
        self._save_error_report(result_dir, net_train_errors, net_test_errors,
                               deePoly_train_errors, deePoly_test_errors, timing_info)
        
        # Plot for each variable
        for var_idx, var_name in enumerate(variables):
            if var_idx >= self.n_eqs:
                print(f"Warning: Variable {var_name} (index {var_idx}) exceeds available equations ({self.n_eqs})")
                continue
                
            # Create training data scatter plot
            self._create_training_scatter_plot(
                x_train, 
                net_train[:, var_idx:var_idx+1], 
                deePoly_train[:, var_idx:var_idx+1], 
                exact_train[:, var_idx:var_idx+1],
                var_name,
                os.path.join(vis_dir, f'training_analysis_{var_name}.png')
            )
            
            # Reconstruct test data grid
            x1_grid, x2_grid, net_test_reshaped = self.reconstruct_grid_test2D(
                data_test, net_test[:, var_idx:var_idx+1], self.config)
            _, _, deePoly_test_reshaped = self.reconstruct_grid_test2D(
                data_test, deePoly_test[:, var_idx:var_idx+1], self.config)
            _, _, exact_test_reshaped = self.reconstruct_grid_test2D(
                data_test, exact_test[:, var_idx:var_idx+1], self.config)
            
            # Create test data grid plot
            self._create_test_grid_plot(
                x1_grid, x2_grid,
                net_test_reshaped[:, :, 0],
                deePoly_test_reshaped[:, :, 0], 
                exact_test_reshaped[:, :, 0],
                var_name,
                os.path.join(vis_dir, f'test_analysis_{var_name}.png')
            )

    def _print_error_statistics(self, net_train_errors, net_test_errors, 
                              deePoly_train_errors, deePoly_test_errors):
        """Print error statistics"""
        print("\n" + "="*80)
        print("ERROR ANALYSIS SUMMARY")
        print("="*80)
        print(f"{'Method':<15} {'Data':<8} {'MSE':<12} {'MAE':<12} {'Max Error':<12} {'Rel Error':<12}")
        print("-"*80)
        print(f"{'PINNs':<15} {'Train':<8} {net_train_errors['MSE']:<12.2e} {net_train_errors['MAE']:<12.2e} {net_train_errors['Max Error']:<12.2e} {net_train_errors['Relative Error']:<12.2e}")
        print(f"{'PINNs':<15} {'Test':<8} {net_test_errors['MSE']:<12.2e} {net_test_errors['MAE']:<12.2e} {net_test_errors['Max Error']:<12.2e} {net_test_errors['Relative Error']:<12.2e}")
        print(f"{'DeePoly':<15} {'Train':<8} {deePoly_train_errors['MSE']:<12.2e} {deePoly_train_errors['MAE']:<12.2e} {deePoly_train_errors['Max Error']:<12.2e} {deePoly_train_errors['Relative Error']:<12.2e}")
        print(f"{'DeePoly':<15} {'Test':<8} {deePoly_test_errors['MSE']:<12.2e} {deePoly_test_errors['MAE']:<12.2e} {deePoly_test_errors['Max Error']:<12.2e} {deePoly_test_errors['Relative Error']:<12.2e}")
        print("="*80)

    def _save_error_report(self, result_dir, net_train_errors, net_test_errors, 
                          deePoly_train_errors, deePoly_test_errors, timing_info=None):
        """Save detailed error report"""
        report_path = os.path.join(result_dir, 'error_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE ERROR ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("CONFIGURATION PARAMETERS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Problem Type: {getattr(self.config, 'problem_type', 'N/A')}\n")
            f.write(f"Method: {getattr(self.config, 'method', 'N/A')}\n")
            f.write(f"Segments: {getattr(self.config, 'n_segments', 'N/A')}\n")
            f.write(f"Polynomial Degree: {getattr(self.config, 'poly_degree', 'N/A')}\n")
            f.write(f"Hidden Dimensions: {getattr(self.config, 'hidden_dims', 'N/A')}\n")
            f.write(f"Domain: {getattr(self.config, 'x_domain', 'N/A')}\n")
            f.write(f"Test Grid: {getattr(self.config, 'points_domain_test', 'N/A')}\n")
            f.write("\n")
            
            # Add timing information if available
            if timing_info:
                f.write("TIMING INFORMATION:\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Solution Time: {timing_info.get('total_time', 'N/A'):.4f} seconds\n")
                f.write(f"Neural Network Training Time (Scoper): {timing_info.get('scoper_time', 'N/A'):.4f} seconds\n")
                f.write(f"Equation Fitting Time (Sniper): {timing_info.get('sniper_time', 'N/A'):.4f} seconds\n")
                
                # Calculate percentages
                total_time = timing_info.get('total_time', 0)
                if total_time > 0:
                    scoper_pct = (timing_info.get('scoper_time', 0) / total_time) * 100
                    sniper_pct = (timing_info.get('sniper_time', 0) / total_time) * 100
                    f.write(f"Scoper Time Percentage: {scoper_pct:.1f}%\n")
                    f.write(f"Sniper Time Percentage: {sniper_pct:.1f}%\n")
                f.write("\n")
            
            f.write("ERROR METRICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Method':<15} {'Dataset':<8} {'MSE':<12} {'MAE':<12} {'Max Error':<12} {'Rel Error':<12}\n")
            f.write("-"*80 + "\n")
            
            # Training errors
            f.write(f"{'PINNs':<15} {'Train':<8} {net_train_errors['MSE']:<12.2e} {net_train_errors['MAE']:<12.2e} {net_train_errors['Max Error']:<12.2e} {net_train_errors['Relative Error']:<12.2e}\n")
            f.write(f"{'DeePoly':<15} {'Train':<8} {deePoly_train_errors['MSE']:<12.2e} {deePoly_train_errors['MAE']:<12.2e} {deePoly_train_errors['Max Error']:<12.2e} {deePoly_train_errors['Relative Error']:<12.2e}\n")
            
            # Test errors
            f.write(f"{'PINNs':<15} {'Test':<8} {net_test_errors['MSE']:<12.2e} {net_test_errors['MAE']:<12.2e} {net_test_errors['Max Error']:<12.2e} {net_test_errors['Relative Error']:<12.2e}\n")
            f.write(f"{'DeePoly':<15} {'Test':<8} {deePoly_test_errors['MSE']:<12.2e} {deePoly_test_errors['MAE']:<12.2e} {deePoly_test_errors['Max Error']:<12.2e} {deePoly_test_errors['Relative Error']:<12.2e}\n")
            
            f.write("\n" + "="*80 + "\n")
            
            # Performance comparison
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-"*40 + "\n")
            train_improvement = ((net_train_errors['MSE'] - deePoly_train_errors['MSE']) / net_train_errors['MSE']) * 100
            test_improvement = ((net_test_errors['MSE'] - deePoly_test_errors['MSE']) / net_test_errors['MSE']) * 100
            
            f.write(f"Training MSE Improvement: {train_improvement:+.2f}%\n")
            f.write(f"Test MSE Improvement: {test_improvement:+.2f}%\n")
            
            if abs(train_improvement) < 5:
                f.write("Training Performance: Similar\n")
            elif train_improvement > 0:
                f.write("Training Performance: DeePoly performs better\n")
            else:
                f.write("Training Performance: PINNs performs better\n")
                
            if abs(test_improvement) < 5:
                f.write("Test Performance: Similar\n")
            elif test_improvement > 0:
                f.write("Test Performance: DeePoly performs better\n")
            else:
                f.write("Test Performance: PINNs performs better\n")
            
            f.write("\n" + "="*80 + "\n")

        print(f"\nDetailed error analysis saved to: {report_path}")

    def _create_training_scatter_plot(self, x_coords, net_sol, deepoly_sol, exact_sol, var_name, save_path):
        """Create training data scatter plot"""
        fig = plt.figure(figsize=(22, 14))
        # Use 12 columns for more precise centering control
        gs = gridspec.GridSpec(2, 12, figure=fig, height_ratios=[2.5, 3], hspace=0.3, wspace=0.25)
        
        # Solution range
        vmin_sol = min(net_sol.min(), deepoly_sol.min(), exact_sol.min())
        vmax_sol = max(net_sol.max(), deepoly_sol.max(), exact_sol.max())
        
        # Error range
        net_error = np.abs(net_sol - exact_sol)
        deepoly_error = np.abs(deepoly_sol - exact_sol)
        vmax_net_err = net_error.max()
        vmax_deepoly_err = deepoly_error.max()
        
        # Get spatial variable names from config
        spatial_vars = getattr(self.config, 'spatial_vars', ['x', 'y'][:self.n_dim])
        x_label = spatial_vars[0] if len(spatial_vars) > 0 else 'x'
        y_label = spatial_vars[1] if len(spatial_vars) > 1 else 'y'
        
        # Top row: Solution distributions (each spans 4 columns)
        ax1 = fig.add_subplot(gs[0, 0:4])
        sc1 = ax1.scatter(x_coords[:, 0], x_coords[:, 1], c=net_sol.flatten(), 
                         cmap='RdYlBu_r', s=25, vmin=vmin_sol, vmax=vmax_sol)
        ax1.set_title(f'PINNs Solution', fontweight='bold', fontsize=16)
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel(y_label, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.8, aspect=20)
        cbar1.set_label(var_name, rotation=0, fontsize=12)
        
        ax2 = fig.add_subplot(gs[0, 4:8])
        sc2 = ax2.scatter(x_coords[:, 0], x_coords[:, 1], c=deepoly_sol.flatten(), 
                         cmap='RdYlBu_r', s=25, vmin=vmin_sol, vmax=vmax_sol)
        ax2.set_title(f'DeePoly Solution', fontweight='bold', fontsize=16)
        ax2.set_xlabel(x_label, fontsize=14)
        ax2.set_ylabel(y_label, fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.8, aspect=20)
        cbar2.set_label(var_name, rotation=0, fontsize=12)
        
        ax3 = fig.add_subplot(gs[0, 8:12])
        sc3 = ax3.scatter(x_coords[:, 0], x_coords[:, 1], c=exact_sol.flatten(), 
                         cmap='RdYlBu_r', s=25, vmin=vmin_sol, vmax=vmax_sol)
        ax3.set_title(f'Exact Solution', fontweight='bold', fontsize=16)
        ax3.set_xlabel(x_label, fontsize=14)
        ax3.set_ylabel(y_label, fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
        cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.8, aspect=20)
        cbar3.set_label(var_name, rotation=0, fontsize=12)
        
        # Bottom row: Error distributions (centered and much larger, with gaps on sides)
        ax4 = fig.add_subplot(gs[1, 1:5])  # Start from column 1, span 4 columns
        sc4 = ax4.scatter(x_coords[:, 0], x_coords[:, 1], c=net_error.flatten(), 
                         cmap='hot', s=45, vmin=0, vmax=vmax_net_err)
        ax4.set_title(f'PINNs Error Distribution (Max: {vmax_net_err:.2e})', 
                      fontweight='bold', fontsize=20)
        ax4.set_xlabel(x_label, fontsize=18)
        ax4.set_ylabel(y_label, fontsize=18)
        ax4.tick_params(labelsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal', adjustable='box')
        cbar4 = plt.colorbar(sc4, ax=ax4, shrink=0.9, aspect=30, pad=0.02)
        cbar4.set_label(f'|Error|', rotation=90, fontsize=16)
        cbar4.ax.tick_params(labelsize=12)
        cbar4.formatter.set_powerlimits((0, 0))
        cbar4.update_ticks()
        
        ax5 = fig.add_subplot(gs[1, 7:11])  # Start from column 7, span 4 columns
        sc5 = ax5.scatter(x_coords[:, 0], x_coords[:, 1], c=deepoly_error.flatten(), 
                         cmap='hot', s=45, vmin=0, vmax=vmax_deepoly_err)
        ax5.set_title(f'DeePoly Error Distribution (Max: {vmax_deepoly_err:.2e})', 
                      fontweight='bold', fontsize=20)
        ax5.set_xlabel(x_label, fontsize=18)
        ax5.set_ylabel(y_label, fontsize=18)
        ax5.tick_params(labelsize=14)
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal', adjustable='box')
        cbar5 = plt.colorbar(sc5, ax=ax5, shrink=0.9, aspect=30, pad=0.02)
        cbar5.set_label(f'|Error|', rotation=90, fontsize=16)
        cbar5.ax.tick_params(labelsize=12)
        cbar5.formatter.set_powerlimits((0, 0))
        cbar5.update_ticks()
        
        plt.suptitle(f'Training Data Analysis - {var_name}', fontsize=22, fontweight='bold', y=0.95)
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _create_test_grid_plot(self, x1_grid, x2_grid, net_sol, deepoly_sol, exact_sol, var_name, save_path):
        """Create test data grid plot"""
        fig = plt.figure(figsize=(22, 14))
        # Use 12 columns for more precise centering control
        gs = gridspec.GridSpec(2, 12, figure=fig, height_ratios=[2.5, 3], hspace=0.3, wspace=0.25)
        
        # Solution range
        vmin_sol = min(net_sol.min(), deepoly_sol.min(), exact_sol.min())
        vmax_sol = max(net_sol.max(), deepoly_sol.max(), exact_sol.max())
        
        # Error range
        net_error = np.abs(net_sol - exact_sol)
        deepoly_error = np.abs(deepoly_sol - exact_sol)
        vmax_net_err = net_error.max()
        vmax_deepoly_err = deepoly_error.max()
        
        # Get spatial variable names from config
        spatial_vars = getattr(self.config, 'spatial_vars', ['x', 'y'][:self.n_dim])
        x_label = spatial_vars[0] if len(spatial_vars) > 0 else 'x'
        y_label = spatial_vars[1] if len(spatial_vars) > 1 else 'y'
        
        # Top row: Solution distributions (each spans 4 columns)
        ax1 = fig.add_subplot(gs[0, 0:4])
        im1 = ax1.contourf(x1_grid, x2_grid, net_sol, levels=20, cmap='RdYlBu_r', 
                          vmin=vmin_sol, vmax=vmax_sol)
        ax1.contour(x1_grid, x2_grid, net_sol, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax1.set_title(f'PINNs Solution', fontweight='bold', fontsize=16)
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel(y_label, fontsize=14)
        ax1.set_aspect('equal', adjustable='box')
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20)
        cbar1.set_label(var_name, rotation=0, fontsize=12)
        
        ax2 = fig.add_subplot(gs[0, 4:8])
        im2 = ax2.contourf(x1_grid, x2_grid, deepoly_sol, levels=20, cmap='RdYlBu_r',
                          vmin=vmin_sol, vmax=vmax_sol)
        ax2.contour(x1_grid, x2_grid, deepoly_sol, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax2.set_title(f'DeePoly Solution', fontweight='bold', fontsize=16)
        ax2.set_xlabel(x_label, fontsize=14)
        ax2.set_ylabel(y_label, fontsize=14)
        ax2.set_aspect('equal', adjustable='box')
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
        cbar2.set_label(var_name, rotation=0, fontsize=12)
        
        ax3 = fig.add_subplot(gs[0, 8:12])
        im3 = ax3.contourf(x1_grid, x2_grid, exact_sol, levels=20, cmap='RdYlBu_r',
                          vmin=vmin_sol, vmax=vmax_sol)
        ax3.contour(x1_grid, x2_grid, exact_sol, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax3.set_title(f'Exact Solution', fontweight='bold', fontsize=16)
        ax3.set_xlabel(x_label, fontsize=14)
        ax3.set_ylabel(y_label, fontsize=14)
        ax3.set_aspect('equal', adjustable='box')
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, aspect=20)
        cbar3.set_label(var_name, rotation=0, fontsize=12)
        
        # Bottom row: Error distributions (centered and much larger, with gaps on sides)
        ax4 = fig.add_subplot(gs[1, 1:5])  # Start from column 1, span 4 columns
        im4 = ax4.contourf(x1_grid, x2_grid, net_error, levels=25, cmap='hot',
                          vmin=0, vmax=vmax_net_err)
        ax4.contour(x1_grid, x2_grid, net_error, levels=12, colors='black', alpha=0.4, linewidths=0.6)
        ax4.set_title(f'PINNs Error Distribution (Max: {vmax_net_err:.2e})', 
                      fontweight='bold', fontsize=20)
        ax4.set_xlabel(x_label, fontsize=18)
        ax4.set_ylabel(y_label, fontsize=18)
        ax4.tick_params(labelsize=14)
        ax4.set_aspect('equal', adjustable='box')
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.9, aspect=30, pad=0.02)
        cbar4.set_label(f'|Error|', rotation=90, fontsize=16)
        cbar4.ax.tick_params(labelsize=12)
        cbar4.formatter.set_powerlimits((0, 0))
        cbar4.update_ticks()
        
        ax5 = fig.add_subplot(gs[1, 7:11])  # Start from column 7, span 4 columns
        im5 = ax5.contourf(x1_grid, x2_grid, deepoly_error, levels=25, cmap='hot',
                          vmin=0, vmax=vmax_deepoly_err)
        ax5.contour(x1_grid, x2_grid, deepoly_error, levels=12, colors='black', alpha=0.4, linewidths=0.6)
        ax5.set_title(f'DeePoly Error Distribution (Max: {vmax_deepoly_err:.2e})', 
                      fontweight='bold', fontsize=20)
        ax5.set_xlabel(x_label, fontsize=18)
        ax5.set_ylabel(y_label, fontsize=18)
        ax5.tick_params(labelsize=14)
        ax5.set_aspect('equal', adjustable='box')
        cbar5 = plt.colorbar(im5, ax=ax5, shrink=0.9, aspect=30, pad=0.02)
        cbar5.set_label(f'|Error|', rotation=90, fontsize=16)
        cbar5.ax.tick_params(labelsize=12)
        cbar5.formatter.set_powerlimits((0, 0))
        cbar5.update_ticks()
        
        plt.suptitle(f'Test Data Analysis - {var_name}', fontsize=22, fontweight='bold', y=0.95)
        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    # 向后兼容的方法
    def reconstruct_grid_test1D(self, data: Dict, preds: np.ndarray, config) -> tuple:
        """1D-specific wrapper"""
        if config.n_dim != 1:
            raise ValueError("reconstruct_grid_test1D only supports 1D data")
        
        coord_grids, u_pred = self.reconstruct_grid_nd(data, preds, config)
        return coord_grids[0], u_pred

    def reconstruct_grid_test3D(self, data: Dict, preds: np.ndarray, config) -> tuple:
        """3D-specific wrapper"""
        if config.n_dim != 3:
            raise ValueError("reconstruct_grid_test3D only supports 3D data")
        
        coord_grids, u_pred = self.reconstruct_grid_nd(data, preds, config)
        return coord_grids[0], coord_grids[1], coord_grids[2], u_pred
