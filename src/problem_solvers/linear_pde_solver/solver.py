from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import os
import json
import importlib.util
import argparse
import time
import sys

from .core import LinearPDENet, LinearPDEFitter
from .utils import LinearPDEConfig, LinearPDEDataGenerator

class LinearPDESolver:

    def __init__(self, config=None, case_dir=None):
        # save case dir
        self.case_dir = case_dir
        self.config = config
        # Prepare data generator and data
        self.datagen = LinearPDEDataGenerator(self.config)
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")

        # Initialize the model
        self.model = LinearPDENet(self.config).to(self.config.device)

        # Load the unified plot module
        self.output_module = self._load_output_module()

    def _load_output_module(self):
        """Load the output.py module from the case directory, which handles both output and visualization"""
        if not self.case_dir:
            raise ValueError("Case path not set, cannot load output.py module")

        output_path = os.path.join(self.case_dir, "output.py")

        if not os.path.exists(output_path):
            raise ValueError(f"Required output.py module not found: {output_path}")

        try:
            print(f"Loading custom output and visualization module: {output_path}")
            # Add the case directory to the Python path
            if self.case_dir not in sys.path:
                sys.path.insert(0, self.case_dir)
            
            spec = importlib.util.spec_from_file_location(
                "custom_output_module", output_path
            )
            output_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(output_module)
            return output_module
        except Exception as e:
            raise RuntimeError(f"Error loading output.py module: {e}")

    def solve(self, result_dir=None):
        """
        Solve the linear partial differential equation problem

        Args:
            result_dir: Directory to save results, if None uses results under the current directory

        Returns:
            tuple: (train_predictions, test_predictions) Predictions for training and test sets
        """
        print("Starting to solve the linear partial differential equation problem...")
        
        # Record total solution time
        solve_start_time = time.time()

        # Determine result directory
        if result_dir is None:
            result_dir = os.path.join(self.case_dir, "results")
        os.makedirs(result_dir, exist_ok=True)
        print(f"Results will be saved in: {os.path.abspath(result_dir)}")

        # Initialize and execute fitting
        print("Initializing fitter...")

        self.fitter = LinearPDEFitter(self.config, self.data_train)
        data_GPU = self.model.prepare_gpu_data(self.data_train)
        
        # Train the network and record final loss
        print("Starting neural network training...")
        self.model.eval()
        self.model.train_net(self.data_train, self.model, data_GPU)
        final_loss = self.model.physics_loss(data_GPU).item()
        print(f"Neural network training completed, final loss: {final_loss:.8e}")
        
        self.fitter.fitter_init(self.model)

        print("Starting equation fitting...")
        start_time = time.time()
        coeffs = self.fitter.fit()
        fit_time = time.time() - start_time
        print(f"Fitting completed, time used: {fit_time:.2f} seconds")

        # Calculate total solution time
        total_solve_time = time.time() - solve_start_time
        print(f"Total solution time: {total_solve_time:.2f} seconds")


        self.output_module.generate_output(self.config,self.data_train,self.data_test,self.fitter,self.model,coeffs,result_dir)


def main():
    """Command line entry function - only for direct testing"""
    # Ensure current path is in Python import path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Specify fixed case path
    case_dir = "cases/linear_pde_cases/test_poisson_2d"
    print(f"Using specified case path: {case_dir}")

    # Construct complete path for configuration file
    config_path = os.path.join(case_dir, "config.json")

    if os.path.exists(config_path):
        # Save current working directory
        original_dir = os.getcwd()

        try:
            # Initialize and run solver
            solver = LinearPDESolver(case_path=case_dir)
            solver.solve()
        finally:
            # Restore original working directory
            os.chdir(original_dir)
    else:
        raise ValueError(f"Configuration file not found: {config_path}")


if __name__ == "__main__":
    # Set __package__ variable to solve relative import issues when running directly
    package_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(package_path, "../../..")))
    __package__ = "src.problem_solvers.linear_pde_solver"

    main() 