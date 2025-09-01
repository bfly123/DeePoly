#!/usr/bin/env python3
"""
Compare reference, training, and test solutions at T=0.1
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

def compare_all_solutions():
    """Compare all solutions at T=0.1"""
    print("=== Comparing All Solutions at T=0.1 ===")
    
    # Load DeePoly solver
    case_dir = os.path.join(current_dir, "cases", "Time_pde_cases", "AC_equation")
    config = TimePDEConfig(case_dir=case_dir)
    solver = TimePDESolver(config)
    
    print("Running DeePoly solver...")
    u_final, u_seg_final, model, coeffs = solver.solve()
    
    # Load reference solution
    ref_solver = ACReferenceSolver(current_dir)
    T = 0.05
    ref_data = ref_solver.solve_reference(T=T, dt=0.1, N_points=512)
    
    # Get training and test data solutions
    train_data = solver.data_train
    test_data = solver.data_test
    
    # Use global solution data which should be correctly ordered
    x_domain = config.x_domain[0]
    x_min, x_max = x_domain[0], x_domain[1]
    
    # For test data, use the global 'x' coordinate which covers full domain
    x_test_global = test_data['x']  # Shape: (100, 1), range [-0.99, 0.99]
    x_test_all = x_test_global * (x_max - x_min) / 2 + (x_max + x_min) / 2  # Convert to physical coords
    x_test_all = x_test_all.flatten()
    
    # For training data, use global 'x' coordinate 
    x_train_global = train_data['x']  # Training global coordinates
    x_train_all = x_train_global * (x_max - x_min) / 2 + (x_max + x_min) / 2
    x_train_all = x_train_all.flatten()
    
    # Get solutions using the final solution from history (which should be at test points)
    if len(solver.solution_history) > 1:
        u_test_final = solver.solution_history[-1]  # Final solution at test points
        u_test_all = u_test_final.flatten()
        print(f"Using final solution: shape={u_test_final.shape}, range=[{u_test_all.min():.6f}, {u_test_all.max():.6f}]")
        
        # For training solution, try construction
        try:
            u_train_constructed, u_train_segments = solver.fitter.construct(train_data, model, coeffs)
            u_train_all = u_train_constructed.flatten()
            print(f"Training construction successful: shape={u_train_constructed.shape}")
        except:
            print("Training construction failed, using global_to_segments conversion")
            # Convert test solution to training points
            u_train_segments = solver.fitter.global_to_segments(u_test_final)
            u_train_all = solver.fitter.segments_to_global(u_train_segments).flatten()
    else:
        print("No solution history available")
        u_test_all = np.zeros_like(x_test_all)
        u_train_all = np.zeros_like(x_train_all)
    
    # Interpolate reference solution to test points for error calculation
    from scipy.interpolate import interp1d
    ref_interpolator = interp1d(ref_data['x'], ref_data['u'], kind='cubic', bounds_error=False, fill_value='extrapolate')
    ref_at_test = ref_interpolator(x_test_all)
    
    # Calculate errors
    test_error = u_test_all - ref_at_test
    test_abs_error = np.abs(test_error)
    test_l2_error = np.linalg.norm(test_error)
    test_rel_l2_error = test_l2_error / np.linalg.norm(ref_at_test)
    
    # Create comparison plot with more subplots for error analysis
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f'Solution Comparison and Error Analysis at T={T:.1f}', fontsize=16)
    
    # Plot 1: All solutions overlaid
    ax1 = axes[0, 0]
    ax1.plot(ref_data['x'], ref_data['u'], 'k-', linewidth=2, label='Reference (Chebfun)', alpha=0.8)
    ax1.scatter(x_train_all, u_train_all, c='red', s=8, alpha=0.6, label='DeePoly (Training)')
    ax1.scatter(x_test_all, u_test_all, c='blue', s=8, alpha=0.6, label='DeePoly (Test)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x,t)')
    ax1.set_title('All Solutions Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reference solution only
    ax2 = axes[0, 1]
    ax2.plot(ref_data['x'], ref_data['u'], 'k-', linewidth=2, label='Reference')
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x,t)')
    ax2.set_title('Reference Solution (Chebfun)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: DeePoly solutions only
    ax3 = axes[1, 0]
    ax3.scatter(x_train_all, u_train_all, c='red', s=8, alpha=0.7, label='Training Points')
    ax3.scatter(x_test_all, u_test_all, c='blue', s=8, alpha=0.7, label='Test Points')
    ax3.set_xlabel('x')
    ax3.set_ylabel('u(x,t)')
    ax3.set_title('DeePoly Solutions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pointwise absolute error
    ax4 = axes[1, 1]
    scatter = ax4.scatter(x_test_all, test_abs_error, c=test_abs_error, s=20, cmap='Reds', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('|u_DeePoly - u_ref|')
    ax4.set_title(f'Pointwise Absolute Error (L∞={np.max(test_abs_error):.3e})')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Absolute Error')
    
    # Plot 5: Pointwise relative error (%) 
    ax5 = axes[2, 0]
    ref_nonzero = np.abs(ref_at_test) > 1e-12  # Avoid division by zero
    rel_error = np.zeros_like(test_error)
    rel_error[ref_nonzero] = 100 * test_abs_error[ref_nonzero] / np.abs(ref_at_test[ref_nonzero])
    rel_error[~ref_nonzero] = 0  # Set to 0 where reference is zero
    
    scatter2 = ax5.scatter(x_test_all, rel_error, c=rel_error, s=20, cmap='Oranges', alpha=0.8)
    ax5.set_xlabel('x')
    ax5.set_ylabel('Relative Error (%)')
    ax5.set_title(f'Pointwise Relative Error (max: {np.max(rel_error):.1f}%)')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax5, label='Relative Error (%)')
    
    # Plot 6: Error statistics and L2 analysis
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    # Calculate statistics
    ref_stats = {
        'min': np.min(ref_data['u']),
        'max': np.max(ref_data['u']),
        'mean': np.mean(ref_data['u']),
        'norm': np.linalg.norm(ref_data['u']),
        'points': len(ref_data['u'])
    }
    
    train_stats = {
        'min': np.min(u_train_all),
        'max': np.max(u_train_all),
        'mean': np.mean(u_train_all),
        'norm': np.linalg.norm(u_train_all),
        'points': len(u_train_all)
    }
    
    test_stats = {
        'min': np.min(u_test_all),
        'max': np.max(u_test_all),
        'mean': np.mean(u_test_all),
        'norm': np.linalg.norm(u_test_all),
        'points': len(u_test_all)
    }
    
    # Calculate error statistics
    error_stats = {
        'l1_error': np.sum(test_abs_error),
        'l2_error': test_l2_error, 
        'linf_error': np.max(test_abs_error),
        'rel_l1': np.sum(test_abs_error) / np.sum(np.abs(ref_at_test)),
        'rel_l2': test_rel_l2_error,
        'rel_linf': np.max(test_abs_error) / np.max(np.abs(ref_at_test)),
        'mean_abs_error': np.mean(test_abs_error),
        'std_abs_error': np.std(test_abs_error),
        'max_rel_error': np.max(rel_error),
        'mean_rel_error': np.mean(rel_error),
        'correlation': np.corrcoef(u_test_all, ref_at_test)[0, 1] if len(u_test_all) > 1 else 0.0
    }

    stats_text = f"""
Solution Statistics at T={T:.1f}:

Reference (Chebfun):
• Min:     {ref_stats['min']:.6f}
• Max:     {ref_stats['max']:.6f}
• Mean:    {ref_stats['mean']:.6f}
• L2 norm: {ref_stats['norm']:.6f}
• Points:  {ref_stats['points']}

DeePoly Test:
• Min:     {test_stats['min']:.6f}
• Max:     {test_stats['max']:.6f}
• Mean:    {test_stats['mean']:.6f}
• L2 norm: {test_stats['norm']:.6f}
• Points:  {test_stats['points']}

Error Analysis:
• L1 Error:        {error_stats['l1_error']:.6e}
• L2 Error:        {error_stats['l2_error']:.6e}
• L∞ Error:        {error_stats['linf_error']:.6e}
• Relative L2:     {error_stats['rel_l2']:.6e}
• Max Rel Error:   {error_stats['max_rel_error']:.2f}%
• Mean Rel Error:  {error_stats['mean_rel_error']:.2f}%
• Correlation:     {error_stats['correlation']:.6f}

Quality Assessment:
"""
    
    # Add quality assessment
    if error_stats['rel_l2'] < 1e-2:
        quality = "EXCELLENT"
    elif error_stats['rel_l2'] < 1e-1:
        quality = "GOOD" 
    elif error_stats['rel_l2'] < 5e-1:
        quality = "FAIR"
    else:
        quality = "POOR"
    
    stats_text += f"• Overall Quality:  {quality}\n"
    stats_text += f"• Domain: [{x_min:.1f}, {x_max:.1f}]"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(case_dir, "results", "all_solutions_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    # Print summary
    print(f"\n=== Solution Comparison Summary ===")
    print(f"Reference solution range: [{ref_stats['min']:.6f}, {ref_stats['max']:.6f}]")
    print(f"DeePoly training range:   [{train_stats['min']:.6f}, {train_stats['max']:.6f}]")
    print(f"DeePoly test range:       [{test_stats['min']:.6f}, {test_stats['max']:.6f}]")
    
    print(f"\n=== Error Analysis Summary ===")
    print(f"L1 Error:         {error_stats['l1_error']:.6e}")
    print(f"L2 Error:         {error_stats['l2_error']:.6e}")
    print(f"L∞ Error:         {error_stats['linf_error']:.6e}")
    print(f"Relative L2:      {error_stats['rel_l2']:.6e}")
    print(f"Max Relative:     {error_stats['max_rel_error']:.2f}%")
    print(f"Mean Relative:    {error_stats['mean_rel_error']:.2f}%")
    print(f"Correlation:      {error_stats['correlation']:.6f}")
    print(f"Quality:          {quality}")
    
    # Assessment
    if error_stats['rel_l2'] < 1e-2:
        print("\n✓ Excellent agreement with reference solution")
    elif error_stats['rel_l2'] < 1e-1:
        print("\n✓ Good agreement with reference solution") 
    elif error_stats['rel_l2'] < 5e-1:
        print("\n⚠ Fair agreement with reference solution")
    else:
        print("\n✗ Poor agreement with reference solution - check implementation")
    
    return ref_data, x_train_all, u_train_all, x_test_all, u_test_all

if __name__ == "__main__":
    try:
        ref_data, x_train, u_train, x_test, u_test = compare_all_solutions()
        print("\n✓ Solution comparison completed successfully!")
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()