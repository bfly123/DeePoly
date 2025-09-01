#!/usr/bin/env python3
"""
Test and compare errors from neural network training step vs fitting step at T=0.1
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
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
from scipy.interpolate import interp1d

def test_training_vs_fitting_errors():
    """Test neural network training error vs fitting error"""
    print("=== Testing Training vs Fitting Errors at T=0.01 ===")
    
    # Load DeePoly solver
    case_dir = os.path.join(current_dir, "cases", "Time_pde_cases", "AC_equation")
    config = TimePDEConfig(case_dir=case_dir)
    solver = TimePDESolver(config)
    
    # Get reference solution
    ref_solver = ACReferenceSolver(current_dir)
    T = 0.01
    ref_data = ref_solver.solve_reference(T=T, dt=0.01, N_points=512)
    
    # Interpolate reference to test points
    test_data = solver.data_test
    x_domain = config.x_domain[0]
    x_min, x_max = x_domain[0], x_domain[1]
    x_test_global = test_data['x']
    x_test_phys = x_test_global * (x_max - x_min) / 2 + (x_max + x_min) / 2
    x_test_phys = x_test_phys.flatten()
    
    ref_interpolator = interp1d(ref_data['x'], ref_data['u'], kind='cubic', bounds_error=False, fill_value='extrapolate')
    ref_at_test = ref_interpolator(x_test_phys)
    
    print("Running solver with intermediate error analysis...")
    
    # Run one time step and capture intermediate results
    dt = config.dt
    T_target = config.T
    
    # Initialize with initial condition from training data
    U = solver.data_train["U"]  # Initial condition
    U_seg = solver.data_train["U_seg"]  # Initial condition in segments
    coeffs = None
    
    # Store results at different stages
    results = {}
    
    print(f"\nStep 0: T = 0.000000, dt = {dt:.6f}")
    
    # Stage 1: Neural network training
    print("  Training neural network for current time step...")
    solver._train_neural_network_step(dt, U_current=U)
    model = solver.model  # Use the trained model from solver
    
    # Get neural network prediction (before polynomial fitting)
    print("  Evaluating neural network prediction...")
    x_test_tensor = torch.tensor(
        test_data["x"], dtype=torch.float64, device=model.config.device, requires_grad=True
    )
    with torch.no_grad():
        model_output = model(x_test_tensor)
        # Handle tuple output (features, prediction) 
        if isinstance(model_output, tuple):
            nn_prediction = model_output[1].cpu().numpy().flatten()  # Use prediction part
        else:
            nn_prediction = model_output.cpu().numpy().flatten()
    
    # Calculate NN error
    nn_error = nn_prediction - ref_at_test
    nn_abs_error = np.abs(nn_error)
    nn_l2_error = np.linalg.norm(nn_error)
    nn_rel_l2_error = nn_l2_error / np.linalg.norm(ref_at_test)
    
    results['neural_network'] = {
        'prediction': nn_prediction,
        'error': nn_error,
        'abs_error': nn_abs_error,
        'l2_error': nn_l2_error,
        'rel_l2_error': nn_rel_l2_error,
        'linf_error': np.max(nn_abs_error)
    }
    
    print(f"  Neural network training completed")
    print(f"  NN L2 error: {nn_l2_error:.6e}, Rel L2: {nn_rel_l2_error:.6e}")
    
    # Stage 2: Polynomial fitting (solving time step)
    print("  Solving polynomial fitting step...")
    # Initialize fitter with trained model
    solver.fitter.fitter_init(model)
    U_new, U_seg_new, coeffs = solver.fitter.solve_time_step(U, U_seg, dt, coeffs_n=coeffs)
    
    # Get final prediction after polynomial fitting at test points
    final_test_prediction, final_test_segments = solver.fitter.construct(
        test_data, model, coeffs
    )
    final_prediction = final_test_prediction.flatten()
    
    # Calculate final error
    final_error = final_prediction - ref_at_test
    final_abs_error = np.abs(final_error)
    final_l2_error = np.linalg.norm(final_error)
    final_rel_l2_error = final_l2_error / np.linalg.norm(ref_at_test)
    
    results['polynomial_fitting'] = {
        'prediction': final_prediction,
        'error': final_error,
        'abs_error': final_abs_error,
        'l2_error': final_l2_error,
        'rel_l2_error': final_rel_l2_error,
        'linf_error': np.max(final_abs_error)
    }
    
    print(f"  Polynomial fitting completed")
    print(f"  Final L2 error: {final_l2_error:.6e}, Rel L2: {final_rel_l2_error:.6e}")
    
    # Calculate improvement from NN to final
    improvement_l2 = nn_l2_error - final_l2_error
    improvement_rel = (nn_rel_l2_error - final_rel_l2_error) / nn_rel_l2_error
    
    results['improvement'] = {
        'l2_improvement': improvement_l2,
        'rel_improvement': improvement_rel,
        'percentage_improvement': improvement_rel * 100
    }
    
    # Print comparison
    print(f"\n=== Error Comparison Summary ===")
    print(f"Neural Network Stage:")
    print(f"  L2 Error:         {nn_l2_error:.6e}")
    print(f"  Relative L2:      {nn_rel_l2_error:.6e} ({nn_rel_l2_error*100:.2f}%)")
    print(f"  L∞ Error:         {results['neural_network']['linf_error']:.6e}")
    
    print(f"\nPolynomial Fitting Stage:")
    print(f"  L2 Error:         {final_l2_error:.6e}")
    print(f"  Relative L2:      {final_rel_l2_error:.6e} ({final_rel_l2_error*100:.2f}%)")
    print(f"  L∞ Error:         {results['polynomial_fitting']['linf_error']:.6e}")
    
    print(f"\nImprovement from NN to Final:")
    print(f"  L2 Improvement:   {improvement_l2:.6e}")
    print(f"  Relative Improvement: {improvement_rel:.1%}")
    
    if improvement_l2 > 0:
        print(f"  ✓ Polynomial fitting improved the solution")
    else:
        print(f"  ⚠ Polynomial fitting did not improve the solution")
    
    # Create detailed visualization
    create_error_comparison_plot(x_test_phys, ref_at_test, results, case_dir)
    
    return results, x_test_phys, ref_at_test

def create_error_comparison_plot(x_test, ref_solution, results, case_dir):
    """Create detailed comparison plot"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Neural Network vs Polynomial Fitting Error Analysis at T=0.01', fontsize=16)
    
    nn_pred = results['neural_network']['prediction']
    final_pred = results['polynomial_fitting']['prediction']
    nn_abs_err = results['neural_network']['abs_error']
    final_abs_err = results['polynomial_fitting']['abs_error']
    
    # Plot 1: All solutions comparison
    ax1 = axes[0, 0]
    ax1.plot(x_test, ref_solution, 'k-', linewidth=2, label='Reference', alpha=0.8)
    ax1.scatter(x_test, nn_pred, c='blue', s=8, alpha=0.7, label='Neural Network')
    ax1.scatter(x_test, final_pred, c='red', s=8, alpha=0.7, label='Final (NN+Poly)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x,t)')
    ax1.set_title('Solution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error comparison
    ax2 = axes[0, 1]
    ax2.scatter(x_test, nn_abs_err, c='blue', s=8, alpha=0.7, label='NN Error')
    ax2.scatter(x_test, final_abs_err, c='red', s=8, alpha=0.7, label='Final Error')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|Error|')
    ax2.set_title('Absolute Error Comparison')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Neural network error details
    ax3 = axes[1, 0]
    scatter1 = ax3.scatter(x_test, nn_abs_err, c=nn_abs_err, s=20, cmap='Blues', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('|u_NN - u_ref|')
    ax3.set_title(f'Neural Network Error (L2={results["neural_network"]["l2_error"]:.3e})')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax3, label='NN Error')
    
    # Plot 4: Final error details
    ax4 = axes[1, 1]
    scatter2 = ax4.scatter(x_test, final_abs_err, c=final_abs_err, s=20, cmap='Reds', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('|u_Final - u_ref|')
    ax4.set_title(f'Final Error (L2={results["polynomial_fitting"]["l2_error"]:.3e})')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax4, label='Final Error')
    
    # Plot 5: Error improvement
    ax5 = axes[2, 0]
    error_improvement = nn_abs_err - final_abs_err
    colors = ['green' if x > 0 else 'orange' for x in error_improvement]
    ax5.scatter(x_test, error_improvement, c=colors, s=15, alpha=0.8)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('x')
    ax5.set_ylabel('Error_NN - Error_Final')
    ax5.set_title('Error Improvement (Green=Better, Orange=Worse)')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Statistics summary
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    # Calculate additional statistics
    nn_stats = results['neural_network']
    final_stats = results['polynomial_fitting']
    improv_stats = results['improvement']
    
    stats_text = f"""
Error Analysis Summary:

Neural Network Stage:
• L2 Error:        {nn_stats['l2_error']:.6e}
• Relative L2:     {nn_stats['rel_l2_error']:.6e} ({nn_stats['rel_l2_error']*100:.2f}%)
• L∞ Error:        {nn_stats['linf_error']:.6e}
• Mean Abs Error:  {np.mean(nn_stats['abs_error']):.6e}

Polynomial Fitting Stage:
• L2 Error:        {final_stats['l2_error']:.6e}
• Relative L2:     {final_stats['rel_l2_error']:.6e} ({final_stats['rel_l2_error']*100:.2f}%)
• L∞ Error:        {final_stats['linf_error']:.6e}
• Mean Abs Error:  {np.mean(final_stats['abs_error']):.6e}

Improvement Analysis:
• L2 Improvement:  {improv_stats['l2_improvement']:.6e}
• Relative Improv: {improv_stats['rel_improvement']:.2%}
• Points Improved: {np.sum(error_improvement > 0)}/{len(error_improvement)}
• Points Worse:    {np.sum(error_improvement < 0)}/{len(error_improvement)}

Quality Assessment:
• NN Quality:      {'GOOD' if nn_stats['rel_l2_error'] < 0.1 else 'FAIR' if nn_stats['rel_l2_error'] < 0.5 else 'POOR'}
• Final Quality:   {'GOOD' if final_stats['rel_l2_error'] < 0.1 else 'FAIR' if final_stats['rel_l2_error'] < 0.5 else 'POOR'}
• Overall Result:  {'IMPROVED' if improv_stats['rel_improvement'] > 0 else 'NO IMPROVEMENT'}
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(case_dir, "results", "training_vs_fitting_error_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Error analysis plot saved to: {output_path}")

if __name__ == "__main__":
    try:
        results, x_test, ref_solution = test_training_vs_fitting_errors()
        print("\n✓ Training vs fitting error analysis completed successfully!")
        
        # Final assessment
        nn_error = results['neural_network']['rel_l2_error']
        final_error = results['polynomial_fitting']['rel_l2_error']
        improvement = results['improvement']['rel_improvement']
        
        print(f"\n=== Final Assessment ===")
        print(f"Neural network provides {nn_error*100:.1f}% error")
        print(f"Final solution provides {final_error*100:.1f}% error")
        if improvement > 0:
            print(f"Polynomial fitting improves accuracy by {improvement:.1%}")
        else:
            print(f"Polynomial fitting does not improve accuracy")
            
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()