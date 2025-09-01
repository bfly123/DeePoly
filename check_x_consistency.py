#!/usr/bin/env python3
"""
Check consistency between test data x positions and reference data x positions
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
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
from ac_reference_solver import ACReferenceSolver

def check_x_consistency():
    """Check x position consistency"""
    print("=== Checking X Position Consistency ===")
    
    # Load DeePoly solver (without running solve)
    case_dir = os.path.join(current_dir, "cases", "Time_pde_cases", "AC_equation")
    config = TimePDEConfig(case_dir=case_dir)
    solver = TimePDESolver(config)
    
    # Get test data (without running solver)
    test_data = solver.data_test
    x_domain = config.x_domain[0]
    x_min, x_max = x_domain[0], x_domain[1]
    
    # DeePoly test data x positions
    x_test_norm = test_data['x']  # Normalized coordinates
    x_test_phys = x_test_norm * (x_max - x_min) / 2 + (x_max + x_min) / 2  # Physical coordinates
    x_test_phys = x_test_phys.flatten()
    
    print(f"DeePoly Test Data:")
    print(f"  Domain: [{x_min}, {x_max}]")
    print(f"  Normalized range: [{x_test_norm.min():.6f}, {x_test_norm.max():.6f}]")
    print(f"  Physical range: [{x_test_phys.min():.6f}, {x_test_phys.max():.6f}]")
    print(f"  Number of points: {len(x_test_phys)}")
    
    # Reference data x positions
    ref_solver = ACReferenceSolver(current_dir)
    T = 0.1
    ref_data = ref_solver.solve_reference(T=T, dt=0.1, N_points=512)
    
    x_ref = ref_data['x']
    print(f"\nReference Data:")
    print(f"  Range: [{x_ref.min():.6f}, {x_ref.max():.6f}]")
    print(f"  Number of points: {len(x_ref)}")
    
    # Check overlap and coverage
    x_test_min, x_test_max = x_test_phys.min(), x_test_phys.max()
    x_ref_min, x_ref_max = x_ref.min(), x_ref.max()
    
    print(f"\nDomain Coverage Analysis:")
    print(f"  DeePoly covers: [{x_test_min:.6f}, {x_test_max:.6f}] (span: {x_test_max - x_test_min:.6f})")
    print(f"  Reference covers: [{x_ref_min:.6f}, {x_ref_max:.6f}] (span: {x_ref_max - x_ref_min:.6f})")
    
    # Check if DeePoly domain is subset of reference domain
    if x_test_min >= x_ref_min and x_test_max <= x_ref_max:
        print("  ✓ DeePoly domain is within reference domain")
        coverage_ratio = (x_test_max - x_test_min) / (x_ref_max - x_ref_min)
        print(f"  DeePoly covers {coverage_ratio:.1%} of reference domain")
    else:
        print("  ✗ DeePoly domain extends beyond reference domain")
    
    # Sample some specific positions
    print(f"\nSample Position Comparison:")
    sample_indices = [0, 25, 50, 75, 99]  # Sample from test data
    for i in sample_indices:
        if i < len(x_test_phys):
            x_test_pos = x_test_phys[i]
            # Find closest reference position
            ref_idx = np.argmin(np.abs(x_ref - x_test_pos))
            x_ref_pos = x_ref[ref_idx]
            diff = abs(x_test_pos - x_ref_pos)
            print(f"  Test[{i:2d}]: {x_test_pos:8.6f} -> Ref[{ref_idx:3d}]: {x_ref_pos:8.6f} (diff: {diff:.6f})")
    
    # Check uniformity
    print(f"\nUniformity Analysis:")
    x_test_sorted = np.sort(x_test_phys)
    test_spacing = np.diff(x_test_sorted)
    print(f"  DeePoly spacing: mean={np.mean(test_spacing):.6f}, std={np.std(test_spacing):.6f}")
    print(f"  DeePoly spacing range: [{np.min(test_spacing):.6f}, {np.max(test_spacing):.6f}]")
    
    ref_spacing = np.diff(x_ref)
    print(f"  Reference spacing: mean={np.mean(ref_spacing):.6f}, std={np.std(ref_spacing):.6f}")
    print(f"  Reference spacing range: [{np.min(ref_spacing):.6f}, {np.max(ref_spacing):.6f}]")
    
    # Check if test data is uniform
    test_uniform_check = np.std(test_spacing) / np.mean(test_spacing)
    ref_uniform_check = np.std(ref_spacing) / np.mean(ref_spacing)
    print(f"  Uniformity coefficient (std/mean):")
    print(f"    DeePoly: {test_uniform_check:.6f} {'(uniform)' if test_uniform_check < 0.01 else '(non-uniform)'}")
    print(f"    Reference: {ref_uniform_check:.6f} {'(uniform)' if ref_uniform_check < 0.01 else '(non-uniform)'}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('X Position Consistency Analysis', fontsize=16)
    
    # Plot 1: Domain coverage comparison
    ax1 = axes[0, 0]
    ax1.axhspan(0.8, 1.2, xmin=(x_test_min - x_ref_min)/(x_ref_max - x_ref_min), 
                xmax=(x_test_max - x_ref_min)/(x_ref_max - x_ref_min), 
                alpha=0.3, color='red', label='DeePoly Domain')
    ax1.axhspan(-0.2, 0.2, xmin=0, xmax=1, alpha=0.3, color='blue', label='Reference Domain')
    ax1.scatter(x_test_phys, np.ones(len(x_test_phys)), c='red', s=20, alpha=0.7, label='DeePoly Points')
    ax1.scatter(x_ref[::10], np.zeros(len(x_ref[::10])), c='blue', s=10, alpha=0.7, label='Reference Points (every 10th)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Domain')
    ax1.set_title('Domain Coverage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Point distribution
    ax2 = axes[0, 1]
    ax2.hist(x_test_phys, bins=20, alpha=0.7, color='red', label='DeePoly', density=True)
    ax2.hist(x_ref, bins=50, alpha=0.5, color='blue', label='Reference', density=True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.set_title('Point Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Spacing analysis
    ax3 = axes[1, 0]
    x_test_mid = (x_test_sorted[:-1] + x_test_sorted[1:]) / 2
    ax3.plot(x_test_mid, test_spacing, 'r-o', markersize=3, label='DeePoly Spacing')
    x_ref_mid = (x_ref[:-1] + x_ref[1:]) / 2
    ax3.plot(x_ref_mid[::10], ref_spacing[::10], 'b-', alpha=0.7, label='Reference Spacing (every 10th)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Δx')
    ax3.set_title('Point Spacing')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
Position Consistency Statistics:

DeePoly Test Data:
• Points:       {len(x_test_phys)}
• Domain:       [{x_test_min:.6f}, {x_test_max:.6f}]
• Span:         {x_test_max - x_test_min:.6f}
• Mean spacing: {np.mean(test_spacing):.6f}
• Uniformity:   {test_uniform_check:.6f}

Reference Data:
• Points:       {len(x_ref)}
• Domain:       [{x_ref_min:.6f}, {x_ref_max:.6f}]
• Span:         {x_ref_max - x_ref_min:.6f}
• Mean spacing: {np.mean(ref_spacing):.6f}
• Uniformity:   {ref_uniform_check:.6f}

Coverage Analysis:
• DeePoly covers {coverage_ratio:.1%} of reference domain
• Domain alignment: {'✓ Good' if x_test_min >= x_ref_min and x_test_max <= x_ref_max else '✗ Poor'}
• Point density ratio: {len(x_test_phys)/len(x_ref):.3f}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(case_dir, "results", "x_position_consistency.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConsistency analysis plot saved to: {output_path}")
    
    # Final assessment
    print(f"\n=== Final Assessment ===")
    if x_test_min >= x_ref_min and x_test_max <= x_ref_max:
        if test_uniform_check < 0.01:
            print("✓ GOOD: DeePoly test points are uniform and within reference domain")
        else:
            print("⚠ FAIR: DeePoly test points are within reference domain but not uniform")
    else:
        print("✗ POOR: DeePoly test points extend beyond reference domain")
    
    print(f"Domain coverage: DeePoly spans {coverage_ratio:.1%} of reference domain")
    print(f"Resolution ratio: DeePoly has {len(x_test_phys)/len(x_ref):.1%} of reference resolution")

if __name__ == "__main__":
    check_x_consistency()