from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import os
import json
import importlib.util
import argparse
import time
import sys

# Use __init__.py to simplify imports
from .core import FuncFittingNet, FuncFittingFitter
from .utils import FuncFittingConfig, FuncFittingDataGenerator
from src.abstract_class.config.base_visualize import BaseVisualizer

class FuncFittingSolver:
    """Function fitting solver"""

    def __init__(self, config=None, case_dir=None):
        # Save case path
        self.case_dir = case_dir
        self.config = config
        # Prepare data generator and data
        self.datagen = FuncFittingDataGenerator(self.config)
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")
        self.visualizer = BaseVisualizer(self.config)

        # Initialize model
        self.model = FuncFittingNet(self.config).to(self.config.device)

        # Load unified plot module
        self.output_module = self._load_output_module()

    def _load_output_module(self):
        """Load output.py module from case directory, which handles both output and visualization"""
        if not self.case_dir:
            raise ValueError("Case path not set, cannot load output.py module")

        output_path = os.path.join(self.case_dir, "output.py")

        if not os.path.exists(output_path):
            raise ValueError(f"Required output.py module not found: {output_path}")

        try:
            print(f"Loading custom output and visualization module: {output_path}")
            # Add case directory to Python path
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
        Solve function fitting problem

        Args:
            result_dir: Directory to save results, if None uses results under current directory

        Returns:
            tuple: (train_predictions, test_predictions) Predictions for training and test sets
        """
        print("Starting to solve function fitting problem...")
        
        # Record total solution time
        solve_start_time = time.time()

        # Determine result directory
        if result_dir is None:
            result_dir = os.path.join(self.case_dir, "results")
        os.makedirs(result_dir, exist_ok=True)
        print(f"Results will be saved in: {os.path.abspath(result_dir)}")

        # Initialize and execute fitting
        print("Initializing fitter...")

        start_time = time.time()
        self.fitter = FuncFittingFitter(self.config, self.data_train)
        data_GPU = self.model.prepare_gpu_data(self.data_train)
        
        # Train network and record final loss
        print("Starting neural network training...")
        self.model.eval()
        self.model = self.model.train_net(self.data_train, self.model, data_GPU)
        final_loss = self.model.physics_loss(data_GPU).item()
        print(f"Neural network training completed, final loss: {final_loss:.8e}")
        scoper_time = time.time() - start_time
        
        self.fitter.fitter_init(self.model)

        print("Starting data fitting...")
        coeffs = self.fitter.fit()
        # Calculate total solution time
        total_time = time.time() - start_time

        sniper_time = total_time - scoper_time

        print(f"Total solution time: {total_time:.2f} seconds")
        print(f"Scoper time: {scoper_time:.2f} seconds")
        print(f"Sniper time: {sniper_time:.2f} seconds")

        # Make predictions
        print("Making predictions on training set...")
        train_predictions, train_segments = self.fitter.construct(
            self.data_train, self.model, coeffs
        )

        print("Making predictions on test set...")
        test_predictions, test_segments = self.fitter.construct(
            self.data_test, self.model, coeffs
        )
        
        # Use unified output module with consistent interface
        print("Generating results using output module...")
        self.output_module.generate_output(
            self.config,
            self.data_train,
            self.data_test,
            self.fitter,
            self.model,
            coeffs,
            result_dir,
            self.visualizer,
            total_time,
            scoper_time,
            sniper_time,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            train_segments=train_segments,
            test_segments=test_segments,
            final_loss=final_loss
        )

        print("Function fitting solution completed!")
        return train_predictions, test_predictions


####### This main is for testing, please use main_solver.py to run the solver##########
def main():
    """Command line entry function - for direct testing only"""
    # Ensure current path is in Python import path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Specify fixed case path
    case_dir = "cases/func_fitting_cases/test_sin"
    print(f"Using specified case path: {case_dir}")

    # Construct complete path for configuration file
    config_path = os.path.join(case_dir, "config.json")

    if os.path.exists(config_path):
        # Save current working directory
        original_dir = os.getcwd()

        try:
            # Initialize and run solver
            solver = FuncFittingSolver(case_path=case_dir)
            solver.solve()
        finally:
            # Restore original working directory
            os.chdir(original_dir)
    else:
        raise ValueError(f"Configuration file not found: {config_path}")


if __name__ == "__main__":
    # Set __package__ variable to resolve relative import issues when running directly
    package_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(package_path, "../../..")))
    __package__ = "src.problem_solvers.func_fitting_solver"

    main()