import numpy as np
import torch
import time
from typing import Dict, Tuple
from torch import optim
import os
import argparse
import sys

# Ensure project modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change relative imports to absolute imports
from src.problem_solvers.time_pde_solver.core.net import TimePDENet

# Import train function
# from src.abstract_class.base_net import train_net
from src.problem_solvers.time_pde_solver.core.fitter import TimePDEFitter
from src.problem_solvers.time_pde_solver.utils.data import TimePDEDataGenerator
from src.problem_solvers.time_pde_solver.utils.visualize import TimePDEVisualizer
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig


class TimePDESolver:
    def __init__(self, config):
        # Initialize config
        self.config = config
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

        u_n, u_n_seg, model, coeffs = self.solve_time_evolution()

        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f} seconds")

        # Evaluate results
        test_predictions, _ = self.fitter.construct(self.data_test, model, coeffs)
        self.visualizer.plot_solution(
            self.data_test, test_predictions, "results/final_solution.png"
        )

        # Save animation with time evolution data
        if hasattr(self, 'time_history') and hasattr(self, 'solution_history'):
            self.visualizer.save_animation(
                "results/flow_evolution.gif", 
                time_history=self.time_history,
                solution_history=self.solution_history,
                data=self.data_test,
                duration=200
            )
        else:
            print("Warning: No time evolution data available for animation")

        return u_n, u_n_seg, model, coeffs

    def solve_time_evolution(
        self,
    ) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """Time evolution solving using IMEX-RK(2,2,2) method"""
        # Initialize time parameters
        it = 0
        T = 0.0
        dt = self.config.dt

        # Initialize solution values directly
        u,u_seg = self.data_train["u"],self.data_train["u_seg"]

        u_test,u_seg_test = self.data_test["u"],self.data_test["u_seg"]

        # Initialize storage for animation data - use test data for visualization
        animation_skip = getattr(self.config, 'animation_skip', 10)  # 每10步保存一次
        self.time_history = [T]
        self.solution_history = []
        
        # Store initial state using test data points
        self.solution_history.append(u_test.copy())

        # Initialize fitter with model
        self.fitter.fitter_init(self.model)

        print("Starting time evolution with IMEX-RK(2,2,2)...")
        self.fitter.print_time_scheme_summary()

        while T < self.config.time:
            # Adaptive time step for first iteration
            if it == 0:
                dt = self.config.dt / 10
            else:
                dt = self.config.dt

            # Estimate stable time step
            if hasattr(self.fitter, "estimate_stable_dt"):
                dt_stable = self.fitter.estimate_stable_dt(u)
                dt = min(dt, dt_stable)

            # Adjust for final time step
            if T + dt > self.config.time:
                dt = self.config.time - T

            print(f"Step {it}: T = {T:.6f}, dt = {dt:.6f}")

            # Execute IMEX-RK time step - direct solution value operation
            u,u_seg,coeffs = self.fitter.solve_time_step(u,u_seg, dt)

            # Update time and iteration
            T += dt
            it += 1

            # Store data for animation using test points (every animation_skip steps)
            if it % animation_skip == 0 or T >= self.config.time:
                self.time_history.append(T)
                # Convert current solution to test data points for visualization
                u_test, _ = self.fitter.construct(self.data_test, self.model, coeffs)
                self.solution_history.append(u_test.copy())

            # Optional: monitoring
            if it % 10 == 0:  # Every 10 steps
                print(f"  Solution norm: {np.linalg.norm(u):.6e}")

        print(f"Time evolution completed. Final time: T = {T:.6f}")
        print(f"Collected {len(self.time_history)} time steps for animation")

        # Convert final solution to coefficients for output compatibility
        #coeffs = self._solution_to_coefficients(u)
        
        # Reconstruct segmented solution for output
        #u_n_seg = self._reconstruct_segmented_solution(u)

        return u, u_seg, self.model, coeffs

  #  def _initialize_solution(self) -> np.ndarray:
  #      """Initialize solution values using initial conditions or neural network prediction"""
  #      try:
  #          # Initialize with small random values or use neural network prediction
  #          total_points = sum(len(seg) for seg in self.data_train["x_segments_norm"])
  #          
  #          # Use neural network to generate initial solution
  #          u_init = np.zeros((total_points, self.config.n_eqs))
  #          
  #          start_idx = 0
  #          for segment_idx in range(len(self.data_train["x_segments_norm"])):
  #              x_seg = self.data_train["x_segments_norm"][segment_idx]
  #              n_points = len(x_seg)
  #              end_idx = start_idx + n_points
  #              
  #              # Use neural network prediction as initial condition
  #              x_tensor = torch.tensor(x_seg, dtype=torch.float64, device=self.config.device)
  #              with torch.no_grad():
  #                  _, u_pred = self.model(x_tensor)
  #                  u_init[start_idx:end_idx, :] = u_pred.cpu().numpy()
  #              
  #              start_idx = end_idx
  #          
  #          print(f"Initialized solution with norm: {np.linalg.norm(u_init):.6e}")
  #          return u_init
  #          
  #      except Exception as e:
  #          print(f"Warning: Failed to initialize with neural network: {e}")
  #          # Fallback: use small random values
  #          total_points = sum(len(seg) for seg in self.data_train["x_segments_norm"])
  #          u_init = np.random.normal(0, 0.01, (total_points, self.config.n_eqs))
  #          return u_init


######## This main is for testing, please use main_solver.py to run the solver##########
#def main():
#    parser = argparse.ArgumentParser(description="Solver entry")
#    parser.add_argument(
#        "--case",
#        type=str,
#        default="time_dependent",
#        choices=["time_dependent"],
#        help="Select the case to run",
#    )
#    args = parser.parse_args()
#
#    if args.case == "time_dependent":
#        config = TimePDEConfig()
#        solver = TimePDESolver(config)
#        solver.solve()
#    else:
#        raise ValueError(f"Unknown case type: {args.case}")
#
#
#if __name__ == "__main__":
#    main()
