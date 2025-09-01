#!/usr/bin/env python3
"""
Debug dimension issues in reference comparison
"""

import numpy as np
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

def debug_data_dimensions():
    """Debug data structure dimensions"""
    print("=== Debugging Data Dimensions ===")
    
    # Load configuration
    case_dir = os.path.join(current_dir, "cases", "Time_pde_cases", "AC_equation")
    config = TimePDEConfig(case_dir=case_dir)
    solver = TimePDESolver(config)
    
    # Run one step to initialize data  
    print("Initializing solver...")
    # Just access the properties to trigger initialization
    _ = solver.data_train
    _ = solver.data_test
    
    # Debug training data structure
    train_data = solver.data_train
    test_data = solver.data_test
    
    print("\n=== Training Data Structure ===")
    print(f"x_segments_norm type: {type(train_data['x_segments_norm'])}")
    print(f"Number of segments: {len(train_data['x_segments_norm'])}")
    
    total_train_points = 0
    for i, x_seg in enumerate(train_data['x_segments_norm']):
        print(f"Segment {i}: shape = {x_seg.shape}, points = {len(x_seg)}")
        total_train_points += len(x_seg)
    print(f"Total training points: {total_train_points}")
    
    print("\n=== Test Data Structure ===")
    if 'x' in test_data:
        print(f"x_test shape: {test_data['x'].shape}")
        print(f"Test points: {len(test_data['x'])}")
    
    print("\n=== Fitter Structure ===")
    print(f"Number of segments (ns): {solver.fitter.ns}")
    print(f"Number of equations (n_eqs): {solver.fitter.config.n_eqs}")
    print(f"Degrees of freedom (dgN): {solver.fitter.dgN}")
    
    # Test solution history format
    print("\n=== Testing Solution Generation ===")
    
    # Create mock coefficients for testing
    mock_coeffs = np.random.randn(solver.fitter.ns, solver.fitter.config.n_eqs, solver.fitter.dgN) * 0.1
    
    try:
        u_pred, u_segments = solver.fitter.construct(train_data, None, mock_coeffs)
        print(f"✓ Construction successful:")
        print(f"  u_pred shape: {u_pred.shape}")
        print(f"  u_segments type: {type(u_segments)}")
        print(f"  u_segments length: {len(u_segments)}")
        for i, seg in enumerate(u_segments):
            print(f"    Segment {i}: shape = {seg.shape}")
            
        # Test global_to_segments and segments_to_global
        u_segs_converted = solver.fitter.global_to_segments(u_pred)
        print(f"\n✓ global_to_segments:")
        print(f"  Input shape: {u_pred.shape}")
        print(f"  Output length: {len(u_segs_converted)}")
        for i, seg in enumerate(u_segs_converted):
            print(f"    Segment {i}: shape = {seg.shape}")
            
        u_global_converted = solver.fitter.segments_to_global(u_segs_converted)
        print(f"\n✓ segments_to_global:")
        print(f"  Output shape: {u_global_converted.shape}")
        
    except Exception as e:
        print(f"✗ Construction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_dimensions()