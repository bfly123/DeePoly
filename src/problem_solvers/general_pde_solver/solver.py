"""
General PDE Solver with Pseudo-Time Stepping

Solves arbitrary steady-state PDEs using pseudo-time stepping iteration.
"""

import numpy as np
import torch
import os
import sys
from typing import Optional

# Ensure project modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.problem_solvers.general_pde_solver.core.fitter import GeneralPDEFitter
from src.problem_solvers.general_pde_solver.core.net import GeneralPDENet
from src.problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.utils.visualize import GeneralPDEVisualizer
from src.problem_solvers.general_pde_solver.utils.data import GeneralPDEDataGenerator


class GeneralPDESolver:
    """General PDE solver using pseudo-time stepping for steady-state problems

    Features:
    - Unified treatment of all derivatives
    - Pseudo-time stepping to steady state
    - Automatic convergence monitoring
    - Adaptive time step control
    """

    def __init__(self, config=None, case_dir=None):
        """Initialize solver"""
        self._initialize_config(config, case_dir)
        self._initialize_components()

    def _initialize_config(self, config, case_dir):
        """Initialize configuration"""
        if config is not None:
            self.config = config
        elif case_dir is not None:
            self.config = GeneralPDEConfig(case_dir=case_dir)
        else:
            raise ValueError("Either config object or case_dir must be provided")

        self.case_dir = case_dir if case_dir else getattr(self.config, "case_dir", None)

        # Export configuration
        if hasattr(self.config, "export_to_json"):
            self.config.export_to_json("config.json")

    def _initialize_components(self):
        """Initialize core components"""
        # Data generator (specialized for general_pde)
        self.datagen = GeneralPDEDataGenerator(self.config)

        # Generate training and test data
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")

        # Neural network model
        self.model = GeneralPDENet(self.config).to(self.config.device)

        # Visualizer for pseudo-time stepping
        self.visualizer = GeneralPDEVisualizer(self.config)

        # General PDE fitter with pseudo-time stepping
        time_scheme = getattr(self.config, 'time_scheme', 'implicit_euler')
        self.fitter = GeneralPDEFitter(
            config=self.config,
            data=self.data_train,
            time_scheme=time_scheme,
            visualizer=self.visualizer
        )

        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)

    def solve(self, U_init: Optional[np.ndarray] = None):
        """Solve steady-state PDE

        Args:
            U_init: Initial guess (optional, defaults to zero)

        Returns:
            Dict containing solution and convergence information
        """
        print("\n" + "=" * 70)
        print("General PDE Solver - Pseudo-Time Stepping")
        print("=" * 70)
        print(f"Problem: {getattr(self.config, 'problem_name', 'Steady-State PDE')}")
        print(f"Domain: {self.config.x_domain}")
        print(f"Segments: {self.config.n_segments}")
        print(f"Polynomial degree: {self.config.poly_degree}")
        print(f"Number of equations: {self.config.n_eqs}")
        print("=" * 70 + "\n")

        # Step 1: Spotter stage (global feature learning)
        print("Step 1: Spotter Stage - Global Feature Learning")
        print("-" * 70)
        U_init = self._spotter_stage()

        # Step 2: Fitter initialization
        print("\nStep 2: Fitter Initialization")
        print("-" * 70)
        self.fitter.fitter_init(model=self.model)
        print("✓ Operators precompiled")
        print("✓ Features generated")

        # Step 3: Pseudo-time stepping to steady state
        print("\nStep 3: Pseudo-Time Stepping to Steady State")
        print("-" * 70)
        U_steady, coeffs_steady, conv_info = self.fitter.solve_steady_state(
            U_init=U_init,
            model=self.model  # Pass model for training at each step
        )

        # Finalize real-time visualization
        if getattr(self.config, 'realtime_visualization', False):
            self.visualizer.finalize_realtime_animation()

        # Step 4: Post-processing and visualization
        if conv_info["converged"]:
            print("\nStep 4: Post-Processing")
            print("-" * 70)
            self._postprocess(U_steady, coeffs_steady, conv_info)

        return {
            "solution": U_steady,
            "coefficients": coeffs_steady,
            "convergence_info": conv_info,
            "config": self.config
        }

    def _spotter_stage(self):
        """Spotter stage: global feature learning via neural network using physics loss

        Returns:
            U_init: Initial solution from Spotter (n_points, n_eqs)
        """
        # Create SpotterNet for initialization (same architecture as GeneralPDENet via BaseNet)
        spotter_model = SpotterNet(self.config, use_periodic=False).to(self.config.device)

        # Get Spotter configuration
        spotter_config = getattr(self.config, 'initial_condition', {}).get('spotter_config', {})
        n_epochs = spotter_config.get('epochs', 1000)
        lr_init = spotter_config.get('learning_rate', 1e-2)
        n_pde_points = spotter_config.get('n_pde_points', 10000)
        print_interval = spotter_config.get('print_interval', 200)

        # Train Spotter using its specialized training method
        print(f"  Training SpotterNet with {n_epochs} epochs...")
        print(f"  PDE sampling points: {n_pde_points} (resampled each epoch)")
        print(f"  Initial learning rate: {lr_init}")

        spotter_model.train_spotter(
            data=self.data_train,
            n_epochs=n_epochs,
            lr_init=lr_init,
            n_pde_points=n_pde_points,
            print_every=print_interval
        )

        # Transfer network parameters from SpotterNet to GeneralPDENet
        print("\n  Transferring network weights from SpotterNet to GeneralPDENet...")
        self.model.load_state_dict(spotter_model.state_dict())
        print("  ✓ Network weights transferred successfully")

        # Extract final prediction as initial guess for pseudo-time stepping
        with torch.no_grad():
            x_train_tensor = torch.tensor(
                self.data_train['x'], dtype=torch.float64,
                device=self.config.device, requires_grad=True
            )
            _, U_pred = self.model(x_train_tensor)
            U_init = U_pred.cpu().numpy()
            u_norm = np.linalg.norm(U_init)
            u_max = np.max(np.abs(U_init))
            print(f"  Initial solution: ||U|| = {u_norm:.6e}, max|U| = {u_max:.6e}")

        return U_init

    def _postprocess(self, U_steady, coeffs_steady, conv_info):
        """Post-process solution and generate visualizations"""
        # All visualization and saving handled by visualizer
        results_dir = self.config.results_dir

        # Save convergence history
        self.visualizer.save_convergence_history(conv_info, results_dir)

        # Visualize final solution
        save_path = os.path.join(results_dir, "final_solution.png")
        self.visualizer.plot_final_solution(U_steady, self.data_train, save_path)

        # Plot residual history
        residual_path = os.path.join(results_dir, "residual_history.png")
        self.visualizer.plot_residual_history(residual_path)

        # Visualize steady-state solution
        self.visualizer.visualize_solution(U_steady, self.data_train, results_dir)

        # Create iteration evolution GIF animation
        if getattr(self.config, 'save_animation', True):
            gif_path = os.path.join(results_dir, "iteration_evolution.gif")
            self.visualizer.create_iteration_gif(gif_path)

        # Save solution data
        self.visualizer.save_solution_data(U_steady, coeffs_steady, conv_info,
                                           self.data_train, results_dir)

        print("✓ Results saved to:", results_dir)


def main():
    """Main entry point for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="General PDE Solver with Pseudo-Time Stepping")
    parser.add_argument("--case_path", type=str, required=True, help="Path to case directory")
    args = parser.parse_args()

    # Create solver
    solver = GeneralPDESolver(case_dir=args.case_path)

    # Solve
    results = solver.solve()

    if results["convergence_info"]["converged"]:
        print("\n✓ Steady-state solution obtained successfully!")
    else:
        print(f"\n✗ Failed to converge: {results['convergence_info']['reason']}")


if __name__ == "__main__":
    main()