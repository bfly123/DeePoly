import argparse
import json
import os
import sys

# Method 1: Get src directory dynamically based on current file location
src_dir = os.path.dirname(os.path.abspath(__file__))

# Add to Python path
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    
# Add project root directory to path
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use simplified imports
from problem_solvers import TimePDESolver, FuncFittingSolver, LinearPDESolver
from problem_solvers.time_pde_solver.utils import TimePDEConfig
from problem_solvers.func_fitting_solver.utils import FuncFittingConfig
from problem_solvers.linear_pde_solver.utils import LinearPDEConfig


def main():
    # Debug mode: directly specify test directory
    DEBUG_MODE = True
   # DEBUG_MODE = False
    
    if DEBUG_MODE:
        # Dynamically construct case path relative to project root
        case_dir = os.path.join(project_root, "cases", "linear_pde_cases", "linear_convection_discontinuity")
        #case_dir = os.path.join(project_root, "cases", "linear_pde_cases", "poisson_2d_sinpixsinpiy")
    else:
        # Get parameters from command line
        parser = argparse.ArgumentParser(description='Solver entry point')
        parser.add_argument('--case_path', type=str, required=True,
                          help='Case directory path, must contain config.json file')
        args = parser.parse_args()
        case_dir = args.case_path
    
    print(f"Using specified case path: {case_dir}")

    # Construct complete path for configuration file
    config_path = os.path.join(case_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Save current working directory
    #original_dir = os.getcwd()

    ## Switch to case directory
    #os.chdir(case_dir)
    #print(f"Current working directory: {os.getcwd()}")

    # Load configuration file
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Get problem type from configuration
    problem_type = config_dict.get("problem_type", None)
    if problem_type is None:
        raise ValueError("Missing 'problem_type' field in configuration file")

    print(f"Detected problem type: {problem_type}")

    # Ensure results directory exists
    os.makedirs(os.path.join(case_dir, "results"), exist_ok=True)

    # Choose solver based on problem type
    if problem_type == "time_pde":
        config = TimePDEConfig(config_path)
        solver = TimePDESolver(config)
        solver.solve()

    elif problem_type == "func_fitting":
        # Create configuration using dataclass
        config = FuncFittingConfig(case_dir=case_dir)
        solver = FuncFittingSolver(config=config, case_dir=case_dir)
        solver.solve()
        
    elif problem_type == "linear_pde":
        # Create configuration using dataclass
        config = LinearPDEConfig(case_dir=case_dir)
        solver = LinearPDESolver(config=config, case_dir=case_dir)
        solver.solve()
        
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    # Restore original working directory
    #os.chdir(original_dir)
    print("Solving completed!")


if __name__ == "__main__":
    main()
