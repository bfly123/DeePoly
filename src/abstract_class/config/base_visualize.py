import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import os
import math
import torch
import matplotlib.gridspec as gridspec

# Set global font and style settings
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.linewidth": 1.2,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,
    }
)


class BaseVisualizer:
    """Base visualization class for DeePoly results analysis"""

    def __init__(self, config):
        self.config = config
        self.n_dim = config.n_dim
        self.n_eqs = config.n_eqs
        self.Ns = np.prod(config.n_segments)

    def _create_figure(self, figsize: tuple = (10, 8)) -> plt.Figure:
        """Create figure with proper settings"""
        return plt.figure(figsize=figsize)

    def _save_figure(self, fig: plt.Figure, save_path: str) -> None:
        """Save figure to file"""
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )

    def _close_figure(self, fig: plt.Figure) -> None:
        """Close figure to free memory"""
        plt.close(fig)

    def calculate_errors(self, pred: np.ndarray, exact: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive error metrics"""
        mse = np.mean((pred - exact) ** 2)
        mae = np.mean(np.abs(pred - exact))
        max_error = np.max(np.abs(pred - exact))
        rel_error = np.mean(np.abs((pred - exact)) / (np.abs(exact) + 1e-12))
        rmse = np.sqrt(mse)

        return {
            "MSE": mse,
            "MAE": mae,
            "Max Error": max_error,
            "Relative Error": rel_error,
            "RMSE": rmse,
        }

    def get_model_predictions(self, model: torch.nn.Module, data: Dict) -> np.ndarray:
        """Get neural network model predictions"""
        x = data["x"]
        # Get model device
        model_device = next(model.parameters()).device

        with torch.no_grad():
            # Create tensor on the same device as the model
            x_tensor = torch.tensor(x, dtype=torch.float64, device=model_device)
            _, predictions = model(x_tensor)

        # Move back to CPU for numpy conversion
        return predictions.cpu().detach().numpy()

    def get_deepoly_predictions(
        self, fitter, data: Dict, model: torch.nn.Module, coeffs: np.ndarray
    ) -> np.ndarray:
        """Get DeePoly hybrid method predictions"""
        predictions, _ = fitter.construct(data, model, coeffs)
        return predictions

    # ======================== 1D Visualization Methods ========================

    def generate_1d_analysis(
        self,
        data_train: Dict,
        data_test: Dict,
        model: torch.nn.Module,
        fitter,
        coeffs: np.ndarray,
        exact_solution_func: Callable,
        result_dir: str,
        variables: Optional[List[str]] = None,
        timing_info: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Generate complete 1D analysis with training and test visualizations

        Args:
            data_train: Training data dictionary
            data_test: Test data dictionary
            model: Trained neural network model
            fitter: DeePoly fitter object
            coeffs: DeePoly coefficients
            exact_solution_func: Function to compute exact solution
            result_dir: Directory to save results
            variables: List of variables to analyze, defaults to all
            timing_info: Dictionary with timing information
        """
        # Default plot all variables
        if variables is None:
            if hasattr(self.config, "vars_list"):
                variables = self.config.vars_list
            else:
                variables = [f"u_{i}" for i in range(self.n_eqs)]

        # Create visualization directory
        vis_dir = os.path.join(result_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Get all prediction results
        analysis_data = self._prepare_1d_analysis_data(
            data_train, data_test, model, fitter, coeffs, exact_solution_func
        )

        # Print and save error statistics
        self._print_error_statistics(**analysis_data["errors"])
        self._save_error_analysis_report(
            result_dir, analysis_data["errors"], timing_info
        )

        # Generate visualizations for each variable
        for var_idx, var_name in enumerate(variables):
            if var_idx >= self.n_eqs:
                print(
                    f"Warning: Variable {var_name} (index {var_idx}) exceeds available equations ({self.n_eqs})"
                )
                continue

            # Create training analysis plot
            self._create_1d_training_scatter_analysis(
                analysis_data,
                var_idx,
                var_name,
                os.path.join(vis_dir, f"training_analysis_{var_name}.png"),
            )

            # Create test analysis plot
            self._create_1d_test_line_analysis(
                analysis_data,
                var_idx,
                var_name,
                os.path.join(vis_dir, f"test_analysis_{var_name}.png"),
            )

    def _prepare_1d_analysis_data(
        self, data_train, data_test, model, fitter, coeffs, exact_solution_func
    ):
        """Prepare all data needed for 1D analysis"""
        # Get coordinates
        x_train = data_train["x"]
        x_test = data_test["x"]

        # Get predictions
        net_train = self.get_model_predictions(model, data_train)
        net_test = self.get_model_predictions(model, data_test)
        exact_train = exact_solution_func(x_train)
        exact_test = exact_solution_func(x_test)
        deepoly_train = self.get_deepoly_predictions(fitter, data_train, model, coeffs)
        deepoly_test = self.get_deepoly_predictions(fitter, data_test, model, coeffs)

        # Calculate errors
        errors = {
            "net_train_errors": self.calculate_errors(net_train, exact_train),
            "net_test_errors": self.calculate_errors(net_test, exact_test),
            "deepoly_train_errors": self.calculate_errors(deepoly_train, exact_train),
            "deepoly_test_errors": self.calculate_errors(deepoly_test, exact_test),
        }

        return {
            "coordinates": {"x_train": x_train, "x_test": x_test},
            "predictions": {
                "net_train": net_train,
                "net_test": net_test,
                "deepoly_train": deepoly_train,
                "deepoly_test": deepoly_test,
                "exact_train": exact_train,
                "exact_test": exact_test,
            },
            "errors": errors,
        }

    def _create_1d_training_scatter_analysis(
        self, analysis_data, var_idx, var_name, save_path
    ):
        """Create 1D training data scatter analysis plot"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(
            2, 2, figure=fig, height_ratios=[3, 2], hspace=0.3, wspace=0.25
        )

        # Extract data for current variable
        x_train = analysis_data["coordinates"]["x_train"].flatten()
        net_sol = analysis_data["predictions"]["net_train"][:, var_idx]
        deepoly_sol = analysis_data["predictions"]["deepoly_train"][:, var_idx]
        exact_sol = analysis_data["predictions"]["exact_train"][:, var_idx]

        # Calculate errors
        net_error = np.abs(net_sol - exact_sol)
        deepoly_error = np.abs(deepoly_sol - exact_sol)

        # Get spatial variable name
        spatial_vars = getattr(self.config, "spatial_vars", ["x"])
        x_label = spatial_vars[0] if len(spatial_vars) > 0 else "x"

        # Top left: Solution comparison with scatter points for training data
        ax1 = fig.add_subplot(gs[0, 0])
        sort_idx = np.argsort(x_train)
        x_sorted = x_train[sort_idx]

        # Plot exact solution as reference line
        ax1.plot(
            x_sorted,
            exact_sol[sort_idx],
            "g-",
            linewidth=3,
            label="Exact Solution",
            alpha=0.9,
            zorder=1,
        )

        # Scatter points for training predictions
        ax1.scatter(
            x_train,
            net_sol,
            c="red",
            s=25,
            alpha=0.7,
            label="PINNs (Training Points)",
            zorder=3,
        )
        ax1.scatter(
            x_train,
            deepoly_sol,
            c="blue",
            s=25,
            alpha=0.7,
            label="DeePoly (Training Points)",
            zorder=2,
            marker="^",
        )

        ax1.set_title(
            f"Training Data: Solutions Comparison - {var_name}",
            fontsize=16,
            fontweight="bold",
        )
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel(var_name, fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Top right: Error scatter plot with color coding
        ax2 = fig.add_subplot(gs[0, 1])
        vmax_err = max(net_error.max(), deepoly_error.max())

        # Create two y-levels for better visualization
        y_net = np.ones_like(x_train) * 0.7
        y_deepoly = np.ones_like(x_train) * 0.3

        sc3 = ax2.scatter(
            x_train,
            y_net,
            c=net_error,
            cmap="Reds",
            s=40,
            vmin=0,
            vmax=vmax_err,
            alpha=0.8,
            label="PINNs Error",
        )
        sc4 = ax2.scatter(
            x_train,
            y_deepoly,
            c=deepoly_error,
            cmap="Blues",
            s=40,
            vmin=0,
            vmax=vmax_err,
            alpha=0.8,
            label="DeePoly Error",
            marker="^",
        )

        ax2.set_title(
            f"Training Data: Error Distribution", fontweight="bold", fontsize=16
        )
        ax2.set_xlabel(x_label, fontsize=14)
        ax2.set_ylabel("Method", fontsize=14)
        ax2.set_yticks([0.3, 0.7])
        ax2.set_yticklabels(["DeePoly", "PINNs"])
        ax2.grid(True, alpha=0.3)

        # Add colorbar
        cbar_net = plt.colorbar(sc3, ax=ax2, shrink=0.4, aspect=15, pad=0.02)
        cbar_net.set_label("|Error|", fontsize=12)
        cbar_net.formatter.set_powerlimits((0, 0))
        cbar_net.update_ticks()

        # Bottom left: Error comparison (log scale)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.semilogy(
            x_sorted,
            net_error[sort_idx],
            "r-",
            linewidth=2.5,
            label="PINNs Error",
            alpha=0.8,
        )
        ax3.semilogy(
            x_sorted,
            deepoly_error[sort_idx],
            "b-",
            linewidth=2.5,
            label="DeePoly Error",
            alpha=0.8,
        )
        ax3.set_title(
            f"Training Data: Error Comparison (Log Scale)",
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_xlabel(x_label, fontsize=12)
        ax3.set_ylabel("|Error| (log)", fontsize=12)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)

        # Bottom right: Error statistics
        ax4 = fig.add_subplot(gs[1, 1])
        methods = ["PINNs", "DeePoly"]
        max_errors = [net_error.max(), deepoly_error.max()]
        mean_errors = [net_error.mean(), deepoly_error.mean()]

        x_pos = np.arange(len(methods))
        width = 0.35

        bars1 = ax4.bar(
            x_pos - width / 2,
            max_errors,
            width,
            label="Max Error",
            alpha=0.8,
            color=["red", "blue"],
        )
        bars2 = ax4.bar(
            x_pos + width / 2,
            mean_errors,
            width,
            label="Mean Error",
            alpha=0.6,
            color=["lightcoral", "lightblue"],
        )

        ax4.set_ylabel("Error Magnitude", fontsize=12)
        ax4.set_title("Training Data: Error Statistics", fontsize=14, fontweight="bold")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(methods)
        ax4.legend(fontsize=11)
        ax4.set_yscale("log")

        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax4.text(
                bar1.get_x() + bar1.get_width() / 2.0,
                height1 * 1.1,
                f"{height1:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=45,
            )
            ax4.text(
                bar2.get_x() + bar2.get_width() / 2.0,
                height2 * 1.1,
                f"{height2:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=45,
            )

        plt.suptitle(
            f"Training Data Analysis - {var_name}",
            fontsize=20,
            fontweight="bold",
            y=0.96,
        )
        try:
            plt.tight_layout()
        except UserWarning:
            pass  # 忽略不兼容的AxesWarning
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _create_1d_test_line_analysis(
        self, analysis_data, var_idx, var_name, save_path
    ):
        """Create 1D test data line analysis plot"""
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(
            2, 3, figure=fig, height_ratios=[2.5, 1.5], hspace=0.3, wspace=0.3
        )

        # Extract data for current variable
        x_test = analysis_data["coordinates"]["x_test"].flatten()
        net_sol = analysis_data["predictions"]["net_test"][:, var_idx]
        deepoly_sol = analysis_data["predictions"]["deepoly_test"][:, var_idx]
        exact_sol = analysis_data["predictions"]["exact_test"][:, var_idx]

        # Sort data for proper line plotting
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        net_sorted = net_sol[sort_idx]
        deepoly_sorted = deepoly_sol[sort_idx]
        exact_sorted = exact_sol[sort_idx]

        # Get spatial variable name
        spatial_vars = getattr(self.config, "spatial_vars", ["x"])
        x_label = spatial_vars[0] if len(spatial_vars) > 0 else "x"

        # Top row: Solution comparisons
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(
            x_sorted, exact_sorted, "g-", linewidth=3, label="Exact Solution", alpha=0.9
        )
        ax1.plot(
            x_sorted, net_sorted, "r--", linewidth=2, label="PINNs Solution", alpha=0.8
        )
        ax1.plot(
            x_sorted,
            deepoly_sorted,
            "b:",
            linewidth=2,
            label="DeePoly Solution",
            alpha=0.8,
        )
        ax1.set_title(
            f"Test Data: Solutions Comparison - {var_name}",
            fontsize=18,
            fontweight="bold",
        )
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel(var_name, fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Bottom row: Error plots
        net_error = np.abs(net_sorted - exact_sorted)
        deepoly_error = np.abs(deepoly_sorted - exact_sorted)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.semilogy(x_sorted, net_error, "r-", linewidth=2, label="PINNs Error")
        ax2.set_title(
            f"PINNs Error (Max: {net_error.max():.2e})", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_ylabel("|Error| (log)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.semilogy(x_sorted, deepoly_error, "b-", linewidth=2, label="DeePoly Error")
        ax3.set_title(
            f"DeePoly Error (Max: {deepoly_error.max():.2e})",
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_xlabel(x_label, fontsize=12)
        ax3.set_ylabel("|Error| (log)", fontsize=12)
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(x_sorted, net_error, "r-", linewidth=2, label="PINNs")
        ax4.plot(x_sorted, deepoly_error, "b-", linewidth=2, label="DeePoly")
        ax4.set_title("Error Comparison (Linear)", fontsize=14, fontweight="bold")
        ax4.set_xlabel(x_label, fontsize=12)
        ax4.set_ylabel("|Error|", fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    # ======================== 2D Visualization Methods ========================

    def generate_2d_analysis(
        self,
        data_train: Dict,
        data_test: Dict,
        model: torch.nn.Module,
        fitter,
        coeffs: np.ndarray,
        exact_solution_func: Callable,
        result_dir: str,
        variables: Optional[List[str]] = None,
        timing_info: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Generate complete 2D analysis with training and test visualizations
        """
        # Default plot all variables
        if variables is None:
            if hasattr(self.config, "vars_list"):
                variables = self.config.vars_list
            else:
                variables = [f"u_{i}" for i in range(self.n_eqs)]

        # Create visualization directory
        vis_dir = os.path.join(result_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Get all prediction results
        analysis_data = self._prepare_2d_analysis_data(
            data_train, data_test, model, fitter, coeffs, exact_solution_func
        )

        # Print and save error statistics
        self._print_error_statistics(**analysis_data["errors"])
        self._save_error_analysis_report(
            result_dir, analysis_data["errors"], timing_info
        )

        # Generate visualizations for each variable
        for var_idx, var_name in enumerate(variables):
            if var_idx >= self.n_eqs:
                print(
                    f"Warning: Variable {var_name} (index {var_idx}) exceeds available equations ({self.n_eqs})"
                )
                continue

            print(f"Generating training analysis for {var_name}...")
            # Create training scatter analysis
            self._create_2d_training_scatter_analysis(
                analysis_data,
                var_idx,
                var_name,
                os.path.join(vis_dir, f"training_analysis_{var_name}.png"),
            )

            print(f"Generating test analysis for {var_name}...")
            # Create test grid analysis
            self._create_2d_test_grid_analysis(
                analysis_data,
                var_idx,
                var_name,
                os.path.join(vis_dir, f"test_analysis_{var_name}.png"),
            )

        print(
            f"2D analysis completed. Generated {len(variables)} sets of training/test visualizations."
        )

    def _prepare_2d_analysis_data(
        self, data_train, data_test, model, fitter, coeffs, exact_solution_func
    ):
        """Prepare all data needed for 2D analysis"""
        # Get coordinates
        x_train = data_train["x"]
        x_test = data_test["x"]

        # Get predictions
        net_train = self.get_model_predictions(model, data_train)
        net_test = self.get_model_predictions(model, data_test)
        exact_train = exact_solution_func(x_train)
        exact_test = exact_solution_func(x_test)
        deepoly_train = self.get_deepoly_predictions(fitter, data_train, model, coeffs)
        deepoly_test = self.get_deepoly_predictions(fitter, data_test, model, coeffs)

        # Calculate errors
        errors = {
            "net_train_errors": self.calculate_errors(net_train, exact_train),
            "net_test_errors": self.calculate_errors(net_test, exact_test),
            "deepoly_train_errors": self.calculate_errors(deepoly_train, exact_train),
            "deepoly_test_errors": self.calculate_errors(deepoly_test, exact_test),
        }

        return {
            "coordinates": {"x_train": x_train, "x_test": x_test},
            "predictions": {
                "net_train": net_train,
                "net_test": net_test,
                "deepoly_train": deepoly_train,
                "deepoly_test": deepoly_test,
                "exact_train": exact_train,
                "exact_test": exact_test,
            },
            "errors": errors,
        }

    def _create_2d_training_scatter_analysis(
        self, analysis_data, var_idx, var_name, save_path
    ):
        """Create 2D training data scatter analysis plot"""
        fig = plt.figure(figsize=(22, 14))
        gs = gridspec.GridSpec(
            2, 12, figure=fig, height_ratios=[2.5, 3], hspace=0.3, wspace=0.25
        )

        # Extract data for current variable
        x_coords = analysis_data["coordinates"]["x_train"]
        net_sol = analysis_data["predictions"]["net_train"][:, var_idx : var_idx + 1]
        deepoly_sol = analysis_data["predictions"]["deepoly_train"][
            :, var_idx : var_idx + 1
        ]
        exact_sol = analysis_data["predictions"]["exact_train"][
            :, var_idx : var_idx + 1
        ]

        # Solution range
        vmin_sol = min(net_sol.min(), deepoly_sol.min(), exact_sol.min())
        vmax_sol = max(net_sol.max(), deepoly_sol.max(), exact_sol.max())

        # Error range
        net_error = np.abs(net_sol - exact_sol)
        deepoly_error = np.abs(deepoly_sol - exact_sol)
        vmax_net_err = net_error.max()
        vmax_deepoly_err = deepoly_error.max()

        # Get spatial variable names
        spatial_vars = getattr(self.config, "spatial_vars", ["x", "y"][: self.n_dim])
        x_label = spatial_vars[0] if len(spatial_vars) > 0 else "x"
        y_label = spatial_vars[1] if len(spatial_vars) > 1 else "y"

        # Top row: Solution distributions (each spans 4 columns)
        ax1 = fig.add_subplot(gs[0, 0:4])
        sc1 = ax1.scatter(
            x_coords[:, 0],
            x_coords[:, 1],
            c=net_sol.flatten(),
            cmap="RdYlBu_r",
            s=25,
            vmin=vmin_sol,
            vmax=vmax_sol,
        )
        ax1.set_title(f"PINNs Solution", fontweight="bold", fontsize=16)
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel(y_label, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal", adjustable="box")
        cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.8, aspect=20)
        cbar1.set_label(var_name, rotation=0, fontsize=12)

        ax2 = fig.add_subplot(gs[0, 4:8])
        sc2 = ax2.scatter(
            x_coords[:, 0],
            x_coords[:, 1],
            c=deepoly_sol.flatten(),
            cmap="RdYlBu_r",
            s=25,
            vmin=vmin_sol,
            vmax=vmax_sol,
        )
        ax2.set_title(f"DeePoly Solution", fontweight="bold", fontsize=16)
        ax2.set_xlabel(x_label, fontsize=14)
        ax2.set_ylabel(y_label, fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal", adjustable="box")
        cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.8, aspect=20)
        cbar2.set_label(var_name, rotation=0, fontsize=12)

        ax3 = fig.add_subplot(gs[0, 8:12])
        sc3 = ax3.scatter(
            x_coords[:, 0],
            x_coords[:, 1],
            c=exact_sol.flatten(),
            cmap="RdYlBu_r",
            s=25,
            vmin=vmin_sol,
            vmax=vmax_sol,
        )
        ax3.set_title(f"Exact Solution", fontweight="bold", fontsize=16)
        ax3.set_xlabel(x_label, fontsize=14)
        ax3.set_ylabel(y_label, fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect("equal", adjustable="box")
        cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.8, aspect=20)
        cbar3.set_label(var_name, rotation=0, fontsize=12)

        # Bottom row: Error distributions (centered and larger)
        ax4 = fig.add_subplot(gs[1, 1:5])
        sc4 = ax4.scatter(
            x_coords[:, 0],
            x_coords[:, 1],
            c=net_error.flatten(),
            cmap="hot",
            s=45,
            vmin=0,
            vmax=vmax_net_err,
        )
        ax4.set_title(
            f"PINNs Error Distribution (Max: {vmax_net_err:.2e})",
            fontweight="bold",
            fontsize=20,
        )
        ax4.set_xlabel(x_label, fontsize=18)
        ax4.set_ylabel(y_label, fontsize=18)
        ax4.tick_params(labelsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect("equal", adjustable="box")
        cbar4 = plt.colorbar(sc4, ax=ax4, shrink=0.9, aspect=30, pad=0.02)
        cbar4.set_label(f"|Error|", rotation=90, fontsize=16)
        cbar4.ax.tick_params(labelsize=12)
        cbar4.formatter.set_powerlimits((0, 0))
        cbar4.update_ticks()

        ax5 = fig.add_subplot(gs[1, 7:11])
        sc5 = ax5.scatter(
            x_coords[:, 0],
            x_coords[:, 1],
            c=deepoly_error.flatten(),
            cmap="hot",
            s=45,
            vmin=0,
            vmax=vmax_deepoly_err,
        )
        ax5.set_title(
            f"DeePoly Error Distribution (Max: {vmax_deepoly_err:.2e})",
            fontweight="bold",
            fontsize=20,
        )
        ax5.set_xlabel(x_label, fontsize=18)
        ax5.set_ylabel(y_label, fontsize=18)
        ax5.tick_params(labelsize=14)
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect("equal", adjustable="box")
        cbar5 = plt.colorbar(sc5, ax=ax5, shrink=0.9, aspect=30, pad=0.02)
        cbar5.set_label(f"|Error|", rotation=90, fontsize=16)
        cbar5.ax.tick_params(labelsize=12)
        cbar5.formatter.set_powerlimits((0, 0))
        cbar5.update_ticks()

        plt.suptitle(
            f"Training Data Analysis - {var_name}",
            fontsize=22,
            fontweight="bold",
            y=0.95,
        )
        try:
            plt.tight_layout()
        except UserWarning:
            pass  # 忽略不兼容的AxesWarning
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _create_2d_test_grid_analysis(
        self, analysis_data, var_idx, var_name, save_path
    ):
        """Create 2D test data grid analysis plot"""
        # Extract data for current variable
        x_coords = analysis_data["coordinates"]["x_test"]
        net_sol = analysis_data["predictions"]["net_test"][:, var_idx : var_idx + 1]
        deepoly_sol = analysis_data["predictions"]["deepoly_test"][
            :, var_idx : var_idx + 1
        ]
        exact_sol = analysis_data["predictions"]["exact_test"][:, var_idx : var_idx + 1]

        # Reconstruct grid from test data
        x1_grid, x2_grid, net_grid = self._reconstruct_test_grid_2d(x_coords, net_sol)
        _, _, deepoly_grid = self._reconstruct_test_grid_2d(x_coords, deepoly_sol)
        _, _, exact_grid = self._reconstruct_test_grid_2d(x_coords, exact_sol)

        # Create the visualization
        fig = plt.figure(figsize=(22, 14))
        gs = gridspec.GridSpec(
            2, 12, figure=fig, height_ratios=[2.5, 3], hspace=0.3, wspace=0.25
        )

        # Solution range for consistent colormaps
        vmin_sol = min(net_grid.min(), deepoly_grid.min(), exact_grid.min())
        vmax_sol = max(net_grid.max(), deepoly_grid.max(), exact_grid.max())

        # Error grids
        net_error_grid = np.abs(net_grid - exact_grid)
        deepoly_error_grid = np.abs(deepoly_grid - exact_grid)
        vmax_net_err = net_error_grid.max()
        vmax_deepoly_err = deepoly_error_grid.max()

        # Get spatial variable names
        spatial_vars = getattr(self.config, "spatial_vars", ["x", "y"][: self.n_dim])
        x_label = spatial_vars[0] if len(spatial_vars) > 0 else "x"
        y_label = spatial_vars[1] if len(spatial_vars) > 1 else "y"

        # Top row: Solution distributions (each spans 4 columns)
        ax1 = fig.add_subplot(gs[0, 0:4])
        im1 = ax1.contourf(
            x1_grid,
            x2_grid,
            net_grid,
            levels=20,
            cmap="RdYlBu_r",
            vmin=vmin_sol,
            vmax=vmax_sol,
        )
        ax1.contour(
            x1_grid,
            x2_grid,
            net_grid,
            levels=10,
            colors="black",
            alpha=0.3,
            linewidths=0.5,
        )
        ax1.set_title(f"PINNs Solution", fontweight="bold", fontsize=16)
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel(y_label, fontsize=14)
        ax1.set_aspect("equal", adjustable="box")
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20)
        cbar1.set_label(var_name, rotation=0, fontsize=12)

        ax2 = fig.add_subplot(gs[0, 4:8])
        im2 = ax2.contourf(
            x1_grid,
            x2_grid,
            deepoly_grid,
            levels=20,
            cmap="RdYlBu_r",
            vmin=vmin_sol,
            vmax=vmax_sol,
        )
        ax2.contour(
            x1_grid,
            x2_grid,
            deepoly_grid,
            levels=10,
            colors="black",
            alpha=0.3,
            linewidths=0.5,
        )
        ax2.set_title(f"DeePoly Solution", fontweight="bold", fontsize=16)
        ax2.set_xlabel(x_label, fontsize=14)
        ax2.set_ylabel(y_label, fontsize=14)
        ax2.set_aspect("equal", adjustable="box")
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
        cbar2.set_label(var_name, rotation=0, fontsize=12)

        ax3 = fig.add_subplot(gs[0, 8:12])
        im3 = ax3.contourf(
            x1_grid,
            x2_grid,
            exact_grid,
            levels=20,
            cmap="RdYlBu_r",
            vmin=vmin_sol,
            vmax=vmax_sol,
        )
        ax3.contour(
            x1_grid,
            x2_grid,
            exact_grid,
            levels=10,
            colors="black",
            alpha=0.3,
            linewidths=0.5,
        )
        ax3.set_title(f"Exact Solution", fontweight="bold", fontsize=16)
        ax3.set_xlabel(x_label, fontsize=14)
        ax3.set_ylabel(y_label, fontsize=14)
        ax3.set_aspect("equal", adjustable="box")
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, aspect=20)
        cbar3.set_label(var_name, rotation=0, fontsize=12)

        # Bottom row: Error distributions (centered and larger)
        ax4 = fig.add_subplot(gs[1, 1:5])
        im4 = ax4.contourf(
            x1_grid,
            x2_grid,
            net_error_grid,
            levels=25,
            cmap="hot",
            vmin=0,
            vmax=vmax_net_err,
        )
        ax4.contour(
            x1_grid,
            x2_grid,
            net_error_grid,
            levels=12,
            colors="black",
            alpha=0.4,
            linewidths=0.6,
        )
        ax4.set_title(
            f"PINNs Error Distribution (Max: {vmax_net_err:.2e})",
            fontweight="bold",
            fontsize=20,
        )
        ax4.set_xlabel(x_label, fontsize=18)
        ax4.set_ylabel(y_label, fontsize=18)
        ax4.tick_params(labelsize=14)
        ax4.set_aspect("equal", adjustable="box")
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.9, aspect=30, pad=0.02)
        cbar4.set_label(f"|Error|", rotation=90, fontsize=16)
        cbar4.ax.tick_params(labelsize=12)
        cbar4.formatter.set_powerlimits((0, 0))
        cbar4.update_ticks()

        ax5 = fig.add_subplot(gs[1, 7:11])
        im5 = ax5.contourf(
            x1_grid,
            x2_grid,
            deepoly_error_grid,
            levels=25,
            cmap="hot",
            vmin=0,
            vmax=vmax_deepoly_err,
        )
        ax5.contour(
            x1_grid,
            x2_grid,
            deepoly_error_grid,
            levels=12,
            colors="black",
            alpha=0.4,
            linewidths=0.6,
        )
        ax5.set_title(
            f"DeePoly Error Distribution (Max: {vmax_deepoly_err:.2e})",
            fontweight="bold",
            fontsize=20,
        )
        ax5.set_xlabel(x_label, fontsize=18)
        ax5.set_ylabel(y_label, fontsize=18)
        ax5.tick_params(labelsize=14)
        ax5.set_aspect("equal", adjustable="box")
        cbar5 = plt.colorbar(im5, ax=ax5, shrink=0.9, aspect=30, pad=0.02)
        cbar5.set_label(f"|Error|", rotation=90, fontsize=16)
        cbar5.ax.tick_params(labelsize=12)
        cbar5.formatter.set_powerlimits((0, 0))
        cbar5.update_ticks()

        plt.suptitle(
            f"Test Data Analysis - {var_name}", fontsize=22, fontweight="bold", y=0.95
        )
        self._save_figure(fig, save_path)
        self._close_figure(fig)

    def _reconstruct_test_grid_2d(self, x_coords, values):
        """Reconstruct 2D grid from test data points"""
        # Get unique coordinates for each dimension
        x1_unique = np.sort(np.unique(x_coords[:, 0]))
        x2_unique = np.sort(np.unique(x_coords[:, 1]))

        # Create meshgrid
        x1_grid, x2_grid = np.meshgrid(x1_unique, x2_unique)

        # Initialize value grid
        value_grid = np.zeros_like(x1_grid)

        # Fill grid by finding matching coordinates
        for i, x1_val in enumerate(x1_unique):
            for j, x2_val in enumerate(x2_unique):
                # Find point with matching coordinates
                mask = (np.abs(x_coords[:, 0] - x1_val) < 1e-10) & (
                    np.abs(x_coords[:, 1] - x2_val) < 1e-10
                )
                if np.any(mask):
                    idx = np.where(mask)[0][0]
                    value_grid[j, i] = values[
                        idx, 0
                    ]  # Note: j,i indexing for proper orientation
                else:
                    # If exact match not found, use nearest neighbor
                    distances = np.sqrt(
                        (x_coords[:, 0] - x1_val) ** 2 + (x_coords[:, 1] - x2_val) ** 2
                    )
                    nearest_idx = np.argmin(distances)
                    value_grid[j, i] = values[nearest_idx, 0]

        return x1_grid, x2_grid, value_grid

    # ======================== Common Methods ========================

    def _print_error_statistics(
        self,
        net_train_errors,
        net_test_errors,
        deepoly_train_errors,
        deepoly_test_errors,
    ):
        """Print comprehensive error statistics"""
        print("\n" + "=" * 80)
        print("ERROR ANALYSIS SUMMARY")
        print("=" * 80)
        print(
            f"{'Method':<15} {'Data':<8} {'MSE':<12} {'MAE':<12} {'Max Error':<12} {'Rel Error':<12}"
        )
        print("-" * 80)
        print(
            f"{'PINNs':<15} {'Train':<8} {net_train_errors['MSE']:<12.2e} {net_train_errors['MAE']:<12.2e} {net_train_errors['Max Error']:<12.2e} {net_train_errors['Relative Error']:<12.2e}"
        )
        print(
            f"{'PINNs':<15} {'Test':<8} {net_test_errors['MSE']:<12.2e} {net_test_errors['MAE']:<12.2e} {net_test_errors['Max Error']:<12.2e} {net_test_errors['Relative Error']:<12.2e}"
        )
        print(
            f"{'DeePoly':<15} {'Train':<8} {deepoly_train_errors['MSE']:<12.2e} {deepoly_train_errors['MAE']:<12.2e} {deepoly_train_errors['Max Error']:<12.2e} {deepoly_train_errors['Relative Error']:<12.2e}"
        )
        print(
            f"{'DeePoly':<15} {'Test':<8} {deepoly_test_errors['MSE']:<12.2e} {deepoly_test_errors['MAE']:<12.2e} {deepoly_test_errors['Max Error']:<12.2e} {deepoly_test_errors['Relative Error']:<12.2e}"
        )
        print("=" * 80)

    def _save_error_analysis_report(self, result_dir, errors, timing_info=None):
        """Save comprehensive error analysis report"""
        report_path = os.path.join(result_dir, "error_analysis_report.txt")

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE ERROR ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("CONFIGURATION PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Problem Type: {getattr(self.config, 'problem_type', 'N/A')}\n")
            f.write(f"Method: {getattr(self.config, 'method', 'N/A')}\n")
            f.write(f"Dimensionality: {self.n_dim}D\n")
            f.write(f"Segments: {getattr(self.config, 'n_segments', 'N/A')}\n")
            f.write(
                f"Polynomial Degree: {getattr(self.config, 'poly_degree', 'N/A')}\n"
            )
            f.write(
                f"Hidden Dimensions: {getattr(self.config, 'hidden_dims', 'N/A')}\n"
            )
            f.write(f"Domain: {getattr(self.config, 'x_domain', 'N/A')}\n")
            f.write(f"Test Grid: {getattr(self.config, 'points_domain_test', 'N/A')}\n")
            f.write("\n")

            # Add timing information if available
            if timing_info:
                f.write("TIMING INFORMATION:\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"Total Solution Time: {timing_info.get('total_time', 'N/A'):.4f} seconds\n"
                )
                f.write(
                    f"Neural Network Training Time (Scoper): {timing_info.get('scoper_time', 'N/A'):.4f} seconds\n"
                )
                f.write(
                    f"Equation Fitting Time (Sniper): {timing_info.get('sniper_time', 'N/A'):.4f} seconds\n"
                )

                # Calculate percentages
                total_time = timing_info.get("total_time", 0)
                if total_time > 0:
                    scoper_pct = (timing_info.get("scoper_time", 0) / total_time) * 100
                    sniper_pct = (timing_info.get("sniper_time", 0) / total_time) * 100
                    f.write(f"Scoper Time Percentage: {scoper_pct:.1f}%\n")
                    f.write(f"Sniper Time Percentage: {sniper_pct:.1f}%\n")
                f.write("\n")

            # Error metrics table
            f.write("ERROR METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"{'Method':<15} {'Dataset':<8} {'MSE':<12} {'MAE':<12} {'Max Error':<12} {'Rel Error':<12}\n"
            )
            f.write("-" * 80 + "\n")

            # Write error data
            for method, dataset, error_data in [
                ("PINNs", "Train", errors["net_train_errors"]),
                ("DeePoly", "Train", errors["deepoly_train_errors"]),
                ("PINNs", "Test", errors["net_test_errors"]),
                ("DeePoly", "Test", errors["deepoly_test_errors"]),
            ]:
                f.write(
                    f"{method:<15} {dataset:<8} {error_data['MSE']:<12.2e} {error_data['MAE']:<12.2e} {error_data['Max Error']:<12.2e} {error_data['Relative Error']:<12.2e}\n"
                )

            f.write("\n" + "=" * 80 + "\n")

            # Performance comparison
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 40 + "\n")
            train_improvement = (
                (
                    errors["net_train_errors"]["MSE"]
                    - errors["deepoly_train_errors"]["MSE"]
                )
                / errors["net_train_errors"]["MSE"]
            ) * 100
            test_improvement = (
                (
                    errors["net_test_errors"]["MSE"]
                    - errors["deepoly_test_errors"]["MSE"]
                )
                / errors["net_test_errors"]["MSE"]
            ) * 100

            f.write(f"Training MSE Improvement: {train_improvement:+.2f}%\n")
            f.write(f"Test MSE Improvement: {test_improvement:+.2f}%\n")

            # Performance assessment
            for dataset, improvement in [
                ("Training", train_improvement),
                ("Test", test_improvement),
            ]:
                if abs(improvement) < 5:
                    f.write(f"{dataset} Performance: Similar\n")
                elif improvement > 0:
                    f.write(f"{dataset} Performance: DeePoly performs better\n")
                else:
                    f.write(f"{dataset} Performance: PINNs performs better\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"\nDetailed error analysis saved to: {report_path}")

    # ======================== Backward Compatibility Methods ========================

    def plot_1d_comparison(self, *args, **kwargs):
        """Backward compatibility wrapper for 1D analysis"""
        return self.generate_1d_analysis(*args, **kwargs)

    def plot_2d_comparison(self, *args, **kwargs):
        """Backward compatibility wrapper for 2D analysis"""
        return self.generate_2d_analysis(*args, **kwargs)

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
        if hasattr(config, "points_domain_test") and isinstance(
            config.points_domain_test, list
        ):
            grid_dims = config.points_domain_test
        else:
            # Fallback: try to infer from data structure
            if "x" in data:
                x_coords = data["x"]
                # For 2D, try to infer grid structure
                if n_dim == 2:
                    x1_unique = np.unique(x_coords[:, 0])
                    x2_unique = np.unique(x_coords[:, 1])
                    grid_dims = [len(x1_unique), len(x2_unique)]
                else:
                    # General fallback
                    total_points = len(x_coords)
                    points_per_dim = int(total_points ** (1.0 / n_dim))
                    grid_dims = [points_per_dim] * n_dim
            else:
                # Use segments to infer
                total_points = sum(
                    len(data["x_segments"][i]) for i in range(total_sections)
                )
                points_per_dim = int(total_points ** (1.0 / n_dim))
                grid_dims = [points_per_dim] * n_dim

        # Extract coordinates based on data structure
        if "x" in data:
            # Direct coordinate array
            all_coords = data["x"]
            all_preds = preds
        else:
            # Segmented data structure
            all_coords = []
            all_preds = []

            pred_idx = 0
            for section in range(total_sections):
                if "x_segments" in data:
                    n_points_section = len(data["x_segments"][section])
                    section_coords = data["x_segments"][section]
                else:
                    n_points_section = len(data["x_segments_norm"][section])
                    section_coords = data["x_segments_norm"][section]

                section_preds = preds[pred_idx : pred_idx + n_points_section, :]

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

            # Update grid dimensions if needed
            if len(unique_dim) != grid_dims[dim]:
                print(
                    f"Info: Dimension {dim} - Adjusted grid size from {grid_dims[dim]} to {len(unique_dim)}"
                )
                grid_dims[dim] = len(unique_dim)

        # Create coordinate grids
        coord_grids = np.meshgrid(*unique_coords, indexing="ij")

        # Initialize prediction grid
        u_pred = np.zeros(tuple(grid_dims) + (n_eqs,))

        # Fill grid by finding matching coordinates
        for i, coord in enumerate(all_coords):
            # Convert coordinates to indices
            indices = []
            for dim in range(n_dim):
                # Find closest match
                distances = np.abs(unique_coords[dim] - coord[dim])
                idx = np.argmin(distances)
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
