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
from scipy.interpolate import interp1d
import scipy.io as sio

def test_training_vs_fitting_errors():
    """Test neural network training error vs fitting error"""
    print("=== Testing Training vs Fitting Errors at T=0.05 ===")
    
    # Load DeePoly solver
    case_dir = os.path.join(current_dir, "cases", "Time_pde_cases", "AC_equation")
    config = TimePDEConfig(case_dir=case_dir)
    solver = TimePDESolver(config)
    
    # Load reference solution from existing MATLAB data
    T = 0.05  # 使用dt=0.05进行测试
    ref_path = os.path.join(case_dir, "reference_data", "allen_cahn_highres.mat")

    print(f"Loading reference solution from: {ref_path}")
    ref_mat = sio.loadmat(ref_path)

    # Extract reference data
    x_ref = ref_mat['x'].flatten()
    t_ref = ref_mat['t'].flatten()
    u_ref_full = ref_mat['usol']  # Shape: (nt, nx)

    print(f"Reference data shape: x={x_ref.shape}, t={t_ref.shape}, u={u_ref_full.shape}")

    # Find closest time to T=0.05
    t_idx = np.argmin(np.abs(t_ref - T))
    t_actual = t_ref[t_idx]
    print(f"Target time T={T:.3f}, actual reference time: {t_actual:.6f}")

    # Get reference solution at closest time
    u_ref_at_t = u_ref_full[t_idx, :]

    ref_data = {
        'x': x_ref,
        'u': u_ref_at_t,
        'is_highres': True,
        'interpolation_method': 'cubic'
    }
    
    # Interpolate reference to training points (not test points!)
    train_data = solver.data_train
    x_domain = config.x_domain[0]
    x_min, x_max = x_domain[0], x_domain[1]
    x_train_global = train_data['x']
    x_train_phys = x_train_global * (x_max - x_min) / 2 + (x_max + x_min) / 2
    x_train_phys = x_train_phys.flatten()
    
    # Use high-precision spline interpolation for reference solution
    from scipy.interpolate import UnivariateSpline
    ref_spline = UnivariateSpline(ref_data['x'], ref_data['u'], s=0)  # s=0 for interpolating spline
    ref_at_train = ref_spline(x_train_phys)
    
    # Print interpolation quality info
    if ref_data.get('is_highres', False):
        print(f"Using high-resolution reference data:")
        print(f"  Reference grid: {len(ref_data['x'])} points")
        print(f"  Training points: {len(x_train_phys)} points")
        print(f"  Interpolation: {ref_data.get('interpolation_method', 'spline')}")
    else:
        print("Warning: Using standard-resolution reference data")
    
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
    solver._train_neural_network_step(it=0, dt=dt, U_current=U)
    model = solver.model  # Use the trained model from solver
    
    # Get neural network prediction (before polynomial fitting) on training points
    print("  Evaluating neural network prediction...")
    x_train_tensor = torch.tensor(
        train_data["x"], dtype=torch.float64, device=model.config.device, requires_grad=True
    )
    with torch.no_grad():
        model_output = model(x_train_tensor)
        # Handle tuple output (features, prediction) 
        if isinstance(model_output, tuple):
            nn_prediction = model_output[1].cpu().numpy().flatten()  # Use prediction part
        else:
            nn_prediction = model_output.cpu().numpy().flatten()
    
    # Calculate NN error on training points
    nn_error = nn_prediction - ref_at_train
    nn_abs_error = np.abs(nn_error)
    nn_l2_error = np.linalg.norm(nn_error)
    nn_rel_l2_error = nn_l2_error / np.linalg.norm(ref_at_train)
    
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
    
    # Use U_new directly (the result from time stepping on training points)
    final_prediction = U_new.flatten()
    
    # Calculate final error on training points using U_new
    final_error = final_prediction - ref_at_train
    final_abs_error = np.abs(final_error)
    final_l2_error = np.linalg.norm(final_error)
    final_rel_l2_error = final_l2_error / np.linalg.norm(ref_at_train)
    
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
    create_error_comparison_plot(x_train_phys, ref_at_train, results, case_dir)
    
    return results, x_train_phys, ref_at_train

def create_error_comparison_plot(x_train, ref_solution, results, case_dir):
    """Create detailed comparison plot"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Neural Network vs U_new Error Analysis on Training Points at T=0.05', fontsize=16)
    
    nn_pred = results['neural_network']['prediction']
    final_pred = results['polynomial_fitting']['prediction']
    nn_abs_err = results['neural_network']['abs_error']
    final_abs_err = results['polynomial_fitting']['abs_error']
    
    # Plot 1: All solutions comparison (all scatter plots for random training points)
    ax1 = axes[0, 0]
    ax1.scatter(x_train, ref_solution, c='black', s=15, alpha=0.8, label='Reference', marker='o')
    ax1.scatter(x_train, nn_pred, c='blue', s=12, alpha=0.7, label='Neural Network', marker='s')
    ax1.scatter(x_train, final_pred, c='red', s=12, alpha=0.7, label='Final (U_new)', marker='^')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x,t)')
    ax1.set_title('Solution Comparison (Random Training Points)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error comparison
    ax2 = axes[0, 1]
    ax2.scatter(x_train, nn_abs_err, c='blue', s=8, alpha=0.7, label='NN Error')
    ax2.scatter(x_train, final_abs_err, c='red', s=8, alpha=0.7, label='U_new Error')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|Error|')
    ax2.set_title('Absolute Error Comparison')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Neural network error details
    ax3 = axes[1, 0]
    scatter1 = ax3.scatter(x_train, nn_abs_err, c=nn_abs_err, s=20, cmap='Blues', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('|u_NN - u_ref|')
    ax3.set_title(f'Neural Network Error (L2={results["neural_network"]["l2_error"]:.3e})')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax3, label='NN Error')
    
    # Plot 4: Final error details
    ax4 = axes[1, 1]
    scatter2 = ax4.scatter(x_train, final_abs_err, c=final_abs_err, s=20, cmap='Reds', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('|u_new - u_ref|')
    ax4.set_title(f'U_new Error (L2={results["polynomial_fitting"]["l2_error"]:.3e})')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax4, label='U_new Error')
    
    # Plot 5: Error improvement
    ax5 = axes[2, 0]
    error_improvement = nn_abs_err - final_abs_err
    colors = ['green' if x > 0 else 'orange' for x in error_improvement]
    ax5.scatter(x_train, error_improvement, c=colors, s=15, alpha=0.8)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('x')
    ax5.set_ylabel('Error_NN - Error_U_new')
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