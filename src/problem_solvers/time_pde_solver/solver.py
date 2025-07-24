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

        # Save animation
        self.visualizer.save_animation("results/flow_evolution.gif", duration=200)

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
        u_current = self._initialize_solution()

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
                dt_stable = self.fitter.estimate_stable_dt(u_current)
                dt = min(dt, dt_stable)

            # Adjust for final time step
            if T + dt > self.config.time:
                dt = self.config.time - T

            print(f"Step {it}: T = {T:.6f}, dt = {dt:.6f}")

            # Execute IMEX-RK time step - direct solution value operation
            u_current = self.fitter.solve_time_step(u_current, dt)

            # Update time and iteration
            T += dt
            it += 1

            # Optional: monitoring
            if it % 10 == 0:  # Every 10 steps
                print(f"  Solution norm: {np.linalg.norm(u_current):.6e}")

        print(f"Time evolution completed. Final time: T = {T:.6f}")

        # Convert final solution to coefficients for output compatibility
        coeffs_final = self._solution_to_coefficients(u_current)
        
        # Reconstruct segmented solution for output
        u_n_seg = self._reconstruct_segmented_solution(u_current)

        return u_current, u_n_seg, self.model, coeffs_final

    def _initialize_solution(self) -> np.ndarray:
        """Initialize solution values using initial conditions or neural network prediction"""
        try:
            # Initialize with small random values or use neural network prediction
            total_points = sum(len(seg) for seg in self.data_train["x_segments_norm"])
            
            # Use neural network to generate initial solution
            u_init = np.zeros((total_points, self.config.n_eqs))
            
            start_idx = 0
            for segment_idx in range(len(self.data_train["x_segments_norm"])):
                x_seg = self.data_train["x_segments_norm"][segment_idx]
                n_points = len(x_seg)
                end_idx = start_idx + n_points
                
                # Use neural network prediction as initial condition
                x_tensor = torch.tensor(x_seg, dtype=torch.float64, device=self.config.device)
                with torch.no_grad():
                    _, u_pred = self.model(x_tensor)
                    u_init[start_idx:end_idx, :] = u_pred.cpu().numpy()
                
                start_idx = end_idx
            
            print(f"Initialized solution with norm: {np.linalg.norm(u_init):.6e}")
            return u_init
            
        except Exception as e:
            print(f"Warning: Failed to initialize with neural network: {e}")
            # Fallback: use small random values
            total_points = sum(len(seg) for seg in self.data_train["x_segments_norm"])
            u_init = np.random.normal(0, 0.01, (total_points, self.config.n_eqs))
            return u_init

    def _reconstruct_segmented_solution(self, u_values: np.ndarray) -> list:
        """Reconstruct segmented solution from global solution values"""
        u_seg_list = []
        start_idx = 0
        
        for segment_idx in range(len(self.data_train["x_segments_norm"])):
            n_points = len(self.data_train["x_segments_norm"][segment_idx])
            end_idx = start_idx + n_points
            
            u_seg = u_values[start_idx:end_idx, :]
            u_seg_list.append(u_seg)
            
            start_idx = end_idx
        
        return u_seg_list

    def _solution_to_coefficients(self, u_values: np.ndarray) -> np.ndarray:
        """Convert solution values back to coefficients using least squares fitting"""
        try:
            coeffs_shape = (self.fitter.ns, self.config.n_eqs, self.fitter.dgN)
            coeffs_new = np.zeros(coeffs_shape)

            # Process each segment
            start_idx = 0
            for segment_idx in range(self.fitter.ns):
                n_points = len(self.data_train["x_segments_norm"][segment_idx])
                end_idx = start_idx + n_points

                # Get segment solution values
                u_seg = u_values[start_idx:end_idx]

                # Get segment features (0th order derivatives)
                features = self.fitter._features[segment_idx][0]

                # Solve least squares for each equation
                for eq_idx in range(self.config.n_eqs):
                    if u_seg.ndim > 1:
                        u_seg_eq = (
                            u_seg[:, eq_idx] if u_seg.shape[1] > eq_idx else u_seg[:, 0]
                        )
                    else:
                        u_seg_eq = u_seg

                    try:
                        # Least squares: features @ coeffs = u_seg_eq
                        coeffs_seg_eq = np.linalg.lstsq(features, u_seg_eq, rcond=None)[
                            0
                        ]
                        # Ensure correct size
                        if len(coeffs_seg_eq) >= self.fitter.dgN:
                            coeffs_new[segment_idx, eq_idx, :] = coeffs_seg_eq[
                                : self.fitter.dgN
                            ]
                        else:
                            coeffs_new[segment_idx, eq_idx, : len(coeffs_seg_eq)] = (
                                coeffs_seg_eq
                            )
                    except:
                        # If least squares fails, use pseudo-inverse
                        try:
                            coeffs_seg_eq = np.linalg.pinv(features) @ u_seg_eq
                            if len(coeffs_seg_eq) >= self.fitter.dgN:
                                coeffs_new[segment_idx, eq_idx, :] = coeffs_seg_eq[
                                    : self.fitter.dgN
                                ]
                            else:
                                coeffs_new[
                                    segment_idx, eq_idx, : len(coeffs_seg_eq)
                                ] = coeffs_seg_eq
                        except:
                            # Final fallback: small random values
                            coeffs_new[segment_idx, eq_idx, :] = np.random.normal(
                                0, 0.01, self.fitter.dgN
                            )

                start_idx = end_idx

            return coeffs_new

        except Exception as e:
            print(f"Warning: Failed to convert solution to coefficients: {e}")
            # Fallback: return zero coefficients
            coeffs_shape = (self.fitter.ns, self.config.n_eqs, self.fitter.dgN)
            return np.zeros(coeffs_shape)


def main():
    parser = argparse.ArgumentParser(description="Solver entry")
    parser.add_argument(
        "--case",
        type=str,
        default="time_dependent",
        choices=["time_dependent"],
        help="Select the case to run",
    )
    args = parser.parse_args()

    if args.case == "time_dependent":
        config = TimePDEConfig()
        solver = TimePDESolver(config)
        solver.solve()
    else:
        raise ValueError(f"Unknown case type: {args.case}")


if __name__ == "__main__":
    main()
