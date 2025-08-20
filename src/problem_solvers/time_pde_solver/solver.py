import numpy as np
import torch
import time
from typing import Dict, Tuple
from torch import optim
import os
import argparse
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
    def __init__(self, config=None, case_dir=None):
        # Initialize config
        if config is not None:
            self.config = config
        elif case_dir is not None:
            self.config = TimePDEConfig(case_dir=case_dir)
        else:
            raise ValueError("Either config object or case_dir must be provided")
        
        # Save case directory
        self.case_dir = case_dir if case_dir is not None else getattr(self.config, 'case_dir', None)
        
        # Export config if it has the method
        if hasattr(self.config, 'export_to_json'):
            self.config.export_to_json("config.json")

        # Initialize data generator
        self.datagen = TimePDEDataGenerator(self.config)

        # Prepare training and test data
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")

        # Initialize model
        self.model = TimePDENet(self.config).to(self.config.device)
        # Initialize fitter
        self.fitter = TimePDEFitter(config=self.config, data=self.data_train)

        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)

        # Initialize visualizer
        self.visualizer = TimePDEVisualizer(self.config)

    def solve(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """Main solving function"""
        start_time = time.time()

        U_n, U_n_seg, model, coeffs = self.solve_time_evolution()

        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f} seconds")

        # Evaluate results
        test_predictions, _ = self.fitter.construct(self.data_test, model, coeffs)
        self.visualizer.plot_solution(
            self.data_test, test_predictions, "results/final_solution.png"
        )

        # Save animation with time evolution data
        if hasattr(self, 'time_history') and hasattr(self, 'solution_history'):
            self.visualizer.save_animation("results/flow_evolution.gif", duration=200)
        else:
            print("Warning: No time evolution data available for animation")

        return U_n, U_n_seg, model, coeffs

    def solve_time_evolution(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """Time evolution solving using IMEX-RK(2,2,2) method"""
        # Initialize time evolution
        it, T, dt, U, U_seg, coeffs = self._initialize_time_evolution()
        
        # Main time stepping loop
        while T < self.config.T:
            # Compute adaptive time step
            dt = self._compute_adaptive_timestep(it, T, dt, U)
            
            print(f"Step {it}: T = {T:.6f}, dt = {dt:.6f}")

            # Train neural network and execute time step
            self._train_neural_network_step(dt, U_current=U)
            U, U_seg, coeffs = self.fitter.solve_time_step(U, U_seg, dt, coeffs_n=coeffs)

            # Update time and iteration
            T += dt
            it += 1

            # Store animation data and monitoring
            self._update_animation_data(it, T, coeffs)
            self._monitor_solution(it, U)

        print(f"Time evolution completed. Final time: T = {T:.6f}")
        print(f"Collected {len(self.time_history)} time steps for animation")

        return U, U_seg, self.model, coeffs

    def _initialize_time_evolution(self) -> Tuple[int, float, float, np.ndarray, list, np.ndarray]:
        """Initialize time evolution parameters and solution"""
        it = 0
        T = 0.0
        dt = self.config.dt

        # Initialize solution values directly
        U, U_seg = self.data_train["U"], self.data_train["U_seg"]
        U_test, U_seg_test = self.data_test["U"], self.data_test["U_seg"]

        # Initialize storage for animation data
        animation_skip = getattr(self.config, 'animation_skip', 10)
        self.time_history = [T]
        self.solution_history = []
        self.solution_history.append(U_test.copy())

        # Initialize fitter with model for operator precompilation
        self.fitter.fitter_init(self.model)
        
        # Initialize coefficients for time evolution
        coeffs = None
        
        return it, T, dt, U, U_seg, coeffs

    def _compute_adaptive_timestep(self, it: int, T: float, dt: float, U: np.ndarray) -> float:
        """Compute adaptive time step based on iteration and stability"""
        # Adaptive time step for first iteration
        if it == 0:
            dt = self.config.dt  #/ 10
        else:
            dt = self.config.dt

        # Estimate stable time step
        if hasattr(self.fitter, "estimate_stable_dt"):
            dt_stable = self.fitter.estimate_stable_dt(U)
            dt = min(dt, dt_stable)

        # Adjust for final time step
        if T + dt > self.config.T:
            dt = self.config.T - T
            
        return dt

    def _update_animation_data(self, it: int, T: float, coeffs: np.ndarray) -> None:
        """Update animation data for visualization"""
        animation_skip = getattr(self.config, 'animation_skip', 10)
        if it % animation_skip == 0 or T >= self.config.T:
            self.time_history.append(T)
            U_test, _ = self.fitter.construct(self.data_test, self.model, coeffs)
            self.solution_history.append(U_test.copy())

    def _monitor_solution(self, it: int, U: np.ndarray) -> None:
        """Monitor solution progress"""
        if it % 10 == 0:  # Every 10 steps
            print(f"  Solution norm: {np.linalg.norm(U):.6e}")

    def _train_neural_network_step(self, dt: float, U_current: np.ndarray = None) -> None:
        """Train neural network for current time step following linear PDE solver pattern"""
        print("  Training neural network for current time step...")
        data_GPU = self.model.prepare_gpu_data(self.data_train,U_current)
               
        self.model.eval()
        # Pass u_n to train_net
        self.model.train_net(self.data_train, self.model, data_GPU, dt=dt)
        final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
        print(f"  Neural network training completed, loss: {final_loss:.8e}")
        self.fitter.fitter_init(self.model)
        
        # Create scatter plot to visualize neural network predictions
        self._plot_neural_network_predictions(dt)

    def _plot_neural_network_predictions(self, dt: float) -> None:
        """Plot neural network predictions as scatter plot"""
        try:
            print("  Creating neural network prediction scatter plot...")
            
            # Get test data points
            x_test = self.data_test["x"].flatten()
            print(f"  Debug: x_test shape: {x_test.shape}, range: [{np.min(x_test):.3f}, {np.max(x_test):.3f}]")
            
            # Get neural network predictions on test data
            x_tensor = torch.tensor(self.data_test["x"], dtype=torch.float64, device=self.config.device)
            x_tensor.requires_grad_(True)  # Enable gradients for x
            print(f"  Debug: x_tensor shape: {x_tensor.shape}")
            
            # Check network parameters first
            total_params = sum(p.numel() for p in self.model.parameters())
            param_norm = sum(p.norm().item() for p in self.model.parameters())
            print(f"  Debug: Total parameters: {total_params}, Parameter norm: {param_norm:.6e}")
            
            # Check input normalization
            print(f"  Debug: x_tensor stats - mean: {torch.mean(x_tensor):.6f}, std: {torch.std(x_tensor):.6f}")
            
            # Don't use no_grad for prediction since we need gradients during forward pass
            self.model.eval()
            features, u_pred = self.model(x_tensor)
            print(f"  Debug: features shape: {features.shape}")
            print(f"  Debug: u_pred shape: {u_pred.shape}")
            print(f"  Debug: u_pred range: [{torch.min(u_pred):.6f}, {torch.max(u_pred):.6f}]")
            print(f"  Debug: features range: [{torch.min(features):.6f}, {torch.max(features):.6f}]")
            
            # Check if u_pred is all zeros or very small
            u_pred_abs_max = torch.max(torch.abs(u_pred))
            print(f"  Debug: max absolute value of u_pred: {u_pred_abs_max:.6e}")
            
            # Check final layer weights
            if hasattr(self.model, 'out'):
                out_weight_norm = torch.norm(self.model.out.weight).item()
                print(f"  Debug: Output layer weight norm: {out_weight_norm:.6e}")
            
            u_pred_np = u_pred.detach().cpu().numpy().flatten()
            
            # Get target/true values if available
            u_true = self.data_test["u"].flatten() if "u" in self.data_test else None
            if u_true is not None:
                print(f"  Debug: u_true shape: {u_true.shape}, range: [{np.min(u_true):.3f}, {np.max(u_true):.3f}]")
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            
            # Plot neural network predictions
            plt.scatter(x_test, u_pred_np, c='red', alpha=0.7, s=30, label='NN Predictions', marker='o')
            
            # Plot true values if available
            if u_true is not None:
                plt.scatter(x_test, u_true, c='blue', alpha=0.5, s=20, label='True Values', marker='.')
            
            # Add labels and formatting
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.title(f'Neural Network Predictions vs True Values (dt={dt:.6f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            results_dir = os.path.join(self.config.results_dir)
            os.makedirs(results_dir, exist_ok=True)
            
            # Create filename with time step info
            if hasattr(self, 'time_history') and len(self.time_history) > 0:
                current_time = self.time_history[-1] if self.time_history else 0.0
                filename = f"nn_predictions_t_{current_time:.6f}_dt_{dt:.6f}.png"
            else:
                filename = f"nn_predictions_dt_{dt:.6f}.png"
            
            output_path = os.path.join(results_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to free memory
            
            print(f"  Neural network prediction plot saved to: {output_path}")
            
            # Print some statistics
            if u_true is not None:
                mse = np.mean((u_pred_np - u_true)**2)
                max_error = np.max(np.abs(u_pred_np - u_true))
                print(f"  MSE between NN and true values: {mse:.6e}")
                print(f"  Max absolute error: {max_error:.6e}")
            
            print(f"  NN prediction range: [{np.min(u_pred_np):.6e}, {np.max(u_pred_np):.6e}]")
            
        except Exception as e:
            print(f"  Warning: Failed to create neural network prediction plot: {e}")


