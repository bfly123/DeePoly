#!/usr/bin/env python3
"""
DeePoly Main Solver Entry Point

This module serves as the main entry point for the DeePoly PDE solver framework.
It handles:
- Dynamic solver selection based on problem type
- Command line argument parsing
- Results directory management
"""

import argparse
import json
import os
import sys

# Setup Python path for imports
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)

# Add directories to Python path
for path in [src_dir, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import problem solvers and configurations
from problem_solvers import TimePDESolver, FuncFittingSolver, LinearPDESolver
from problem_solvers.time_pde_solver.utils import TimePDEConfig
from problem_solvers.func_fitting_solver.utils import FuncFittingConfig
from problem_solvers.linear_pde_solver.utils import LinearPDEConfig
from meta_coding.auto_code_manager import AutoCodeManager


def handle_autocode(case_dir: str) -> bool:
    """
    Handle automatic code generation and consistency checking

    Args:
        case_dir: Directory containing the case configuration

    Returns:
        True if process was restarted, False otherwise
    """
    manager = AutoCodeManager(case_dir)
    return manager.handle_autocode_workflow()


def create_solver(problem_type: str, case_dir: str):
    """
    Create and run the appropriate solver based on problem type

    Args:
        problem_type: Type of problem (time_pde, func_fitting, linear_pde)
        case_dir: Directory containing the case configuration
    """
    if problem_type == "time_pde":
        config = TimePDEConfig(case_dir=case_dir)
        solver = TimePDESolver(config)
        solver.solve()

    elif problem_type == "func_fitting":
        config = FuncFittingConfig(case_dir=case_dir)
        solver = FuncFittingSolver(config=config, case_dir=case_dir)
        solver.solve()

    elif problem_type == "linear_pde":
        config = LinearPDEConfig(case_dir=case_dir)
        solver = LinearPDESolver(config=config, case_dir=case_dir)
        solver.solve()

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='DeePoly PDE Solver - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --case_path cases/time_pde_cases/Allen_Cahn/AC_equation_100_0.1
  %(prog)s --case_path cases/linear_pde_cases/poisson_2d
  %(prog)s --case_path cases/func_fitting_cases/case_2d
        """
    )

    parser.add_argument(
        '--case_path',
        type=str,
        required=True,
        help='Path to case directory containing config.json'
    )

    return parser.parse_args()


def main():
    """Main execution function"""

    # Configuration for debug/development mode
    DEBUG_MODE = True  # Set to False for production use

    if DEBUG_MODE:
        # Use predefined case for debugging
        # Uncomment the case you want to test:

        # case_dir = os.path.join(project_root, "cases", "linear_pde_cases", "linear_convection_discontinuity")
        # case_dir = os.path.join(project_root, "cases", "Time_pde_cases", "Burgers", "Burgers1")
        # case_dir = os.path.join(project_root, "cases", "Time_pde_cases", "KDV_equation")
        case_dir = os.path.join(project_root, "cases", "Time_pde_cases", "Allen_Cahn", "AC_equation_100_0.1")
        # case_dir = os.path.join(project_root, "cases", "linear_pde_cases", "poisson_2d_sinpixsinpiy")
    else:
        # Parse command line arguments
        args = parse_arguments()
        case_dir = args.case_path

    print(f"Using case directory: {case_dir}")

    # Validate configuration file exists
    config_path = os.path.join(case_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Handle automatic code generation and restart if needed
    if handle_autocode(case_dir):
        return  # Process was restarted

    # Load configuration to determine problem type
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    problem_type = config_dict.get("problem_type", None)
    if problem_type is None:
        raise ValueError("Missing 'problem_type' field in configuration file")

    print(f"Problem type: {problem_type}")

    # Ensure results directory exists
    results_dir = os.path.join(case_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create and run the appropriate solver
    create_solver(problem_type, case_dir)

    print("Solving completed successfully!")


if __name__ == "__main__":
    main()