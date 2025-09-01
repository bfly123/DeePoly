#!/usr/bin/env python3
"""
Debug data structure to understand the issue
"""

import numpy as np
import os
import sys

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.problem_solvers.time_pde_solver.solver import TimePDESolver
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig

def debug_data_structure():
    """Debug the data structure"""
    print("=== Debugging Data Structure ===")
    
    # Load DeePoly solver
    case_dir = os.path.join(current_dir, "cases", "Time_pde_cases", "AC_equation")
    config = TimePDEConfig(case_dir=case_dir)
    solver = TimePDESolver(config)
    
    print("Running DeePoly solver...")
    u_final, u_seg_final, model, coeffs = solver.solve()
    
    # Analyze data structures
    train_data = solver.data_train
    test_data = solver.data_test
    
    print("\n=== Training Data Structure ===")
    print(f"Keys: {list(train_data.keys())}")
    if 'x_segments_norm' in train_data:
        x_train_segments = train_data['x_segments_norm']
        print(f"Number of training segments: {len(x_train_segments)}")
        for i, x_seg in enumerate(x_train_segments):
            print(f"  Segment {i}: shape={x_seg.shape}, range=[{x_seg.min():.6f}, {x_seg.max():.6f}]")
    
    print(f"\n=== Test Data Structure ===")
    print(f"Keys: {list(test_data.keys())}")
    if 'x_segments_norm' in test_data:
        x_test_segments = test_data['x_segments_norm']
        print(f"Number of test segments: {len(x_test_segments)}")
        for i, x_seg in enumerate(x_test_segments):
            print(f"  Segment {i}: shape={x_seg.shape}, range=[{x_seg.min():.6f}, {x_seg.max():.6f}]")
    
    if 'x' in test_data:
        print(f"Test data 'x' key: shape={test_data['x'].shape}, range=[{test_data['x'].min():.6f}, {test_data['x'].max():.6f}]")
    
    print(f"\n=== Solution History ===")
    print(f"Number of time steps: {len(solver.solution_history)}")
    for i, sol in enumerate(solver.solution_history):
        print(f"  Step {i}: shape={sol.shape}, range=[{sol.min():.6f}, {sol.max():.6f}]")
    
    print(f"\n=== Config Domain ===")
    print(f"x_domain: {config.x_domain}")
    
    # Try construction
    print(f"\n=== Construction Test ===")
    try:
        u_train_final, u_train_segments = solver.fitter.construct(train_data, model, coeffs)
        print(f"Training construction successful:")
        print(f"  u_train_final shape: {u_train_final.shape}")
        print(f"  u_train_segments length: {len(u_train_segments)}")
        for i, u_seg in enumerate(u_train_segments):
            print(f"    Segment {i}: shape={u_seg.shape}, range=[{u_seg.min():.6f}, {u_seg.max():.6f}]")
    except Exception as e:
        print(f"Training construction failed: {e}")
    
    try:
        u_test_final, u_test_segments = solver.fitter.construct(test_data, model, coeffs)
        print(f"Test construction successful:")
        print(f"  u_test_final shape: {u_test_final.shape}")
        print(f"  u_test_segments length: {len(u_test_segments)}")
        for i, u_seg in enumerate(u_test_segments):
            print(f"    Segment {i}: shape={u_seg.shape}, range=[{u_seg.min():.6f}, {u_seg.max():.6f}]")
    except Exception as e:
        print(f"Test construction failed: {e}")
    
    return solver, train_data, test_data

if __name__ == "__main__":
    solver, train_data, test_data = debug_data_structure()