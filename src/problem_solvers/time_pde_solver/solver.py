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
#from src.abstract_class.base_net import train_net
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
        self.model = TimePDENet(
            in_dim=2, hidden_dims=self.config.hidden_dims, out_dim=self.config.n_eqs
        ).to(self.config.device)

        # Prepare GPU data
        self.data_GPU = self.datagen.prepare_gpu_data(self.data_train)

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
        self.visualizer.plot_solution(self.data_test, test_predictions, "results/final_solution.png")

        # Save animation
        self.visualizer.save_animation("results/flow_evolution.gif", duration=200)

        return u_n, u_n_seg, model, coeffs

    def solve_time_evolution(
        self,
    ) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """Time evolution solving, using TR-BDF2 format"""
        # Initialize time step parameters
        it = 0
        T = 0
        dt = self.config.dt

        # Initialize solution and segmented solution
        u_n = self.data_train["u"]
        u_n_seg = self.data_train["u_segments"]
        f_n = None  # Store spatial derivative term

        while T < self.config.time:
            # Adjust time step
            if it == 0:
                dt = self.config.dt / 10
            else:
                dt = self.config.dt

            it += 1
            T += dt

            if T + dt > self.config.time:
                dt = self.config.time - T
            print(f"T = {T:.3f}")

            if it == 1:
                # First step using first-order method
                u_np1, u_np1_seg, f_np1, f_np1_seg, self.model, coeffs = (
                    self._time_evolve(
                        "1st_order",
                        {
                            "u_n": u_n,
                            "u_ng": None,
                            "f_n": None,
                        },{
                            "u_n_seg": u_n_seg,
                            "u_ng_seg": None,
                            "f_n_seg": None,
                        },
                        dt,
                    )
                )
            else:
                # TR-BDF2 first stage: calculate u^{n+γ}
                u_np1, u_np1_seg, f_np1, f_np1_seg, self.model, coeffs = (
                    self._time_evolve(
                        "pre",
                        {
                            "u_n": u_n,
                            "u_ng": None,
                            "f_n": f_n,
                        },
                        {
                            "u_n_seg": u_n_seg,
                            "u_ng_seg": None,
                            "f_n_seg": f_n_seg,
                        },
                        dt,
                    )
                )

            # Update solution
            u_n = u_np1
            u_n_seg = u_np1_seg
            f_n = f_np1
            f_n_seg = f_np1_seg
            
            # Visualize current time step
            self.visualizer.plot_evolution_step(
                T, u_n, self.data_train["x"],
                f"{self.config.results_dir}/evolution/t_{T:.3f}.png"
            )

        self.visualizer.close_evolution_plot()
        return u_n, u_n_seg, self.model, coeffs

    def _time_evolve(
        self, step: str, data: Dict, data_seg: Dict, dt: float
    ) -> Tuple[np.ndarray, list, np.ndarray, list, torch.nn.Module, np.ndarray]:
        """Single time step evolution"""
        # Train model
        train(data, self.model, self.data_GPU, self.config, optim, step=step, dt=dt)
        self.fitter.fitter_init(self.model)

        # Fit and predict
        coeffs = self.fitter.fit(data_seg, step=step, dt=dt)
        u, u_seg = self.fitter.construct(self.data_train, self.model, coeffs)

        # Calculate spatial derivative
        u_x, u_x_seg = self.fitter.construct(
            self.data_train, self.model, coeffs, [1, 0]
        )
        u_y, u_y_seg = self.fitter.construct(
            self.data_train, self.model, coeffs, [0, 1]
        )

        # Calculate total spatial derivative term
        f = u_x + u_y
        f_seg = [x_seg + y_seg for x_seg, y_seg in zip(u_x_seg, u_y_seg)]

        return u, u_seg, f, f_seg, self.model, coeffs


def main():
    parser = argparse.ArgumentParser(description='Solver entry')
    parser.add_argument('--case', type=str, default='time_dependent',
                      choices=['time_dependent'],
                      help='Select the case to run')
    args = parser.parse_args()

    if args.case == 'time_dependent':
        config = TimePDEConfig()
        solver = TimePDESolver(config)
        solver.solve()
    else:
        raise ValueError(f"Unknown case type: {args.case}")


if __name__ == "__main__":
    main() 