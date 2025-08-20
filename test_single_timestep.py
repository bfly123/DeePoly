#!/usr/bin/env python3
"""
单时间步测试脚本
Test script for single time step execution and visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import sys
import json

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.problem_solvers.time_pde_solver.solver import TimePDESolver
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig

def test_single_timestep():
    """Run single timestep test"""
    print("=== Single Timestep Test Started ===")
    
    # Set case directory
    case_dir = os.path.join(current_dir, "cases", "Time_pde_cases", "AC_equation")
    print(f"Case directory: {case_dir}")
    
    # Check configuration file
    config_path = os.path.join(case_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load and display configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    print(f"Time domain: T = {config_dict['T']}, dt = {config_dict['dt']}")
    print(f"Spatial domain: {config_dict['x_domain']}")
    print(f"Initial condition: {config_dict['Initial_conditions'][0]['value']}")
    
    # Create configuration and solver
    config = TimePDEConfig(case_dir=case_dir)
    solver = TimePDESolver(config)
    
    print("\n=== Starting Time Evolution Solver ===")
    
    # Run solver
    u_final, u_seg_final, model, coeffs = solver.solve()
    
    print(f"\n=== Solving Completed ===")
    print(f"Final solution dimension: {u_final.shape}")
    print(f"Time history points: {len(solver.time_history)}")
    
    # Generate detailed visualization
    create_detailed_visualization(solver, config, u_final)
    
    return solver, u_final, model, coeffs

def create_detailed_visualization(solver, config, u_final):
    """Create detailed visualization charts"""
    print("\n=== Generating Visualization Charts ===")
    
    # Create charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Allen-Cahn Equation Single Timestep Test Results', fontsize=16)
    
    # 1. Initial condition visualization
    ax1 = axes[0, 0]
    x_test = solver.data_test["x"]
    u_initial = solver.solution_history[0]
    
    ax1.plot(x_test.flatten(), u_initial.flatten(), 'b-', linewidth=2, label='Initial Condition')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x,0)')
    ax1.set_title('Initial Condition: $u_0(x) = x^2 \cos(\pi x)$')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Final state visualization
    ax2 = axes[0, 1]
    try:
        u_test_final, _ = solver.fitter.construct(solver.data_test, solver.model, coeffs)
    except:
        # If construction fails, use last history record
        u_test_final = solver.solution_history[-1] if len(solver.solution_history) > 1 else u_initial
    
    ax2.plot(x_test.flatten(), u_initial.flatten(), 'b--', alpha=0.6, label=f't=0')
    ax2.plot(x_test.flatten(), u_test_final.flatten(), 'r-', linewidth=2, label=f't={solver.time_history[-1]:.3f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x,t)')
    ax2.set_title('Solution Evolution Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Time evolution history
    ax3 = axes[1, 0]
    if len(solver.solution_history) > 1:
        # Calculate L2 norm change over time
        norms = [np.linalg.norm(sol) for sol in solver.solution_history]
        ax3.plot(solver.time_history, norms, 'g-o', linewidth=2, markersize=4)
        ax3.set_xlabel('Time t')
        ax3.set_ylabel('||u||_2')
        ax3.set_title('L2 Norm Time Evolution')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient time history data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Time Evolution (Insufficient Data)')
    
    # 4. Spatiotemporal evolution plot
    ax4 = axes[1, 1]
    if len(solver.solution_history) > 1:
        # Create spatiotemporal grid
        X, T = np.meshgrid(x_test.flatten(), solver.time_history)
        U = np.array([sol.flatten() for sol in solver.solution_history])
        
        contour = ax4.contourf(X, T, U, levels=20, cmap='RdYlBu')
        ax4.set_xlabel('Space x')
        ax4.set_ylabel('Time t')
        ax4.set_title('Spatiotemporal Evolution')
        plt.colorbar(contour, ax=ax4, label='u(x,t)')
    else:
        ax4.text(0.5, 0.5, 'Insufficient time history data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Spatiotemporal Evolution (Insufficient Data)')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save figure
    results_dir = os.path.join(config.case_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, "single_timestep_test.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    
    # Don't display chart, only save
    # plt.show()  # Commented out for non-interactive mode
    
    # Print numerical statistics
    print(f"\n=== Numerical Statistics ===")
    print(f"Initial solution norm: {np.linalg.norm(u_initial):.6e}")
    print(f"Final solution norm: {np.linalg.norm(u_test_final):.6e}")
    print(f"Solution change magnitude: {np.linalg.norm(u_test_final - u_initial):.6e}")
    print(f"Max value change: {np.max(u_test_final) - np.max(u_initial):.6e}")
    print(f"Min value change: {np.min(u_test_final) - np.min(u_initial):.6e}")

if __name__ == "__main__":
    try:
        solver, u_final, model, coeffs = test_single_timestep()
        print("\n✓ Single timestep test completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()