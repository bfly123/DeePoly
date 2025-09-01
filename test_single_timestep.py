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
from ac_reference_solver import ACReferenceSolver

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
    
    # Generate reference solution comparison
    compare_with_reference(solver, config, u_final)
    
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

def compare_with_reference(solver, config, u_final=None):
    """Compare DeePoly solution with Chebfun reference solution"""
    print("\n=== Reference Solution Comparison ===")
    
    try:
        # Initialize reference solver
        ref_solver = ACReferenceSolver(current_dir)
        
        # Get parameters from config
        T = solver.time_history[-1] if len(solver.time_history) > 0 else config.dt
        dt = config.dt
        
        # Load reference solution for single timestep
        print(f"Loading reference solution for single timestep T={T:.6f}")
        ref_data = ref_solver.solve_reference(T=T, dt=dt, N_points=512)
        
        # Use data_test for uniform distribution comparison
        test_data = solver.data_test
        print(f"Using data_test for uniform distribution comparison")
        
        # Get final DeePoly solution at test points (uniform distribution)
        try:
            # For single timestep, use the latest coefficients with test data
            if hasattr(solver, 'coeffs') and solver.coeffs is not None:
                u_deepoly_final, u_segments_final = solver.fitter.construct(test_data, solver.model, solver.coeffs)
                print(f"Using constructed solution at test points with {len(u_segments_final)} segments")
            else:
                raise AttributeError("No coefficients available")
        except Exception as e:
            print(f"Construction failed ({e}), using solution history")
            # For single timestep, use the final solution in history
            if len(solver.solution_history) > 1:
                u_global_final = solver.solution_history[-1]
                u_segments_final = solver.fitter.global_to_segments(u_global_final)
                print(f"Using final solution from history with {len(u_segments_final)} segments")
            else:
                # Fallback to initial condition (should not happen in normal case)
                u_global_initial = solver.solution_history[0] 
                u_segments_final = solver.fitter.global_to_segments(u_global_initial)
                print(f"Warning: Using initial condition with {len(u_segments_final)} segments")
        
        # Check consistency between test data and reference data domains
        test_x_segments = test_data.get('x_segments_norm', test_data.get('x', []))
        if len(test_x_segments) > 0:
            # Convert normalized test coordinates to physical coordinates  
            x_domain = getattr(config, 'x_domain', [[-1, 1]])
            x_min, x_max = x_domain[0][0], x_domain[0][1]
            
            # Collect all test points
            x_test_all = []
            for x_seg_norm in test_x_segments:
                # Convert normalized to physical coordinates
                x_seg_phys = x_seg_norm * (x_max - x_min) / 2 + (x_max + x_min) / 2
                x_test_all.extend(x_seg_phys.flatten())
            x_test_all = np.array(x_test_all)
            
            print(f"Test data points: {len(x_test_all)} in domain [{x_test_all.min():.6f}, {x_test_all.max():.6f}]")
            print(f"Reference data points: {len(ref_data['x'])} in domain [{ref_data['x'].min():.6f}, {ref_data['x'].max():.6f}]")
        
        # Prepare test data dictionary for comparison
        test_data_dict = {
            'x_segments_norm': test_x_segments,
            'x_domain': getattr(config, 'x_domain', [[-1, 1]])
        }
        
        # Use comprehensive test data comparison  
        comparison_result = ref_solver.compare_with_training_data(
            test_data_dict, u_segments_final, ref_data, T
        )
        
        metrics = comparison_result['global_metrics']
        segment_metrics = comparison_result['segment_metrics']
        
        # Create comparison visualization using test data points (uniform distribution)
        results_dir = os.path.join(config.case_dir, "results")
        comparison_path = os.path.join(results_dir, "reference_comparison_test.png")
        
        # Create visualization with test data
        create_training_data_comparison_plot(
            test_data_dict, u_segments_final, comparison_result, comparison_path
        )
        
        # Print comprehensive comparison metrics
        print(f"\n=== Accuracy Metrics vs Reference Solution ===")
        print(f"Test Points:      {metrics['n_total_points']}")
        print(f"Segments:         {metrics['n_segments']}")
        print(f"L1 Error:         {metrics['L1_error']:.6e}")
        print(f"L2 Error:         {metrics['L2_error']:.6e}")
        print(f"L∞ Error:         {metrics['Linf_error']:.6e}")
        print(f"Relative L1:      {metrics['relative_L1']:.6e}")
        print(f"Relative L2:      {metrics['relative_L2']:.6e}")
        print(f"Relative L∞:      {metrics['relative_Linf']:.6e}")
        print(f"Correlation:      {metrics['correlation']:.6f}")
        print(f"Reference Norm:   {metrics['reference_norm']:.6e}")
        print(f"DeePoly Norm:     {metrics['deepoly_norm']:.6e}")
        
        # Print segment-wise metrics
        print(f"\n=== Segment-wise Error Analysis ===")
        for seg_metric in segment_metrics:
            print(f"Segment {seg_metric['segment_idx']:2d}: L2={seg_metric['L2_error']:.4e}, L∞={seg_metric['Linf_error']:.4e}, Points={seg_metric['n_points']} (DP:{seg_metric.get('deepoly_points', 'N/A')}, Ref:{seg_metric.get('ref_points', 'N/A')})")
        
        # Assessment
        if metrics['relative_L2'] < 1e-2:
            print("\n✓ Excellent agreement with reference solution")
        elif metrics['relative_L2'] < 1e-1:
            print("\n✓ Good agreement with reference solution")
        elif metrics['relative_L2'] < 5e-1:
            print("\n⚠ Moderate agreement with reference solution")
        else:
            print("\n✗ Poor agreement with reference solution - check implementation")
            
    except Exception as e:
        print(f"⚠ Reference solution comparison failed: {e}")
        print("  This is normal if MATLAB or Chebfun is not available")
        print("  DeePoly solution analysis will continue without reference comparison")

def create_training_data_comparison_plot(test_data, deepoly_segments, comparison_result, output_path):
    """Create detailed comparison plot using test data points (uniform distribution)"""
    print(f"Creating test data comparison plot: {output_path}")
    # Note: deepoly_segments parameter kept for API consistency but data comes from comparison_result
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DeePoly vs Reference Solution at Test Points (Uniform Distribution)', fontsize=16)
        
        metrics = comparison_result['global_metrics']
        segment_metrics = comparison_result['segment_metrics'] 
        ref_at_train = comparison_result['reference_at_training']
        
        # 1. Global solution comparison (scatter plot for test points)
        ax1 = axes[0, 0]
        x_global = np.concatenate([x_seg.flatten() for x_seg in test_data['x_segments_norm']])
        deepoly_global = comparison_result['deepoly_global']
        ref_global = ref_at_train['U_global_ref'].flatten()
        
        ax1.scatter(x_global, deepoly_global, c='red', s=4, alpha=0.7, label='DeePoly')
        ax1.scatter(x_global, ref_global, c='blue', s=4, alpha=0.7, label='Reference')
        ax1.set_xlabel('x (test points, uniform)')
        ax1.set_ylabel('u(x,t)')
        ax1.set_title('Global Solution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Absolute error at test points
        ax2 = axes[0, 1]
        absolute_error = np.abs(deepoly_global - ref_global)
        scatter = ax2.scatter(x_global, absolute_error, c=absolute_error, s=6, cmap='Reds')
        ax2.set_xlabel('x (test points, uniform)')
        ax2.set_ylabel('|u_DeePoly - u_ref|')
        ax2.set_title(f'Absolute Error (L∞={metrics["Linf_error"]:.2e})')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Error')
        
        # 3. Segment-wise error distribution
        ax3 = axes[0, 2]
        seg_indices = [m['segment_idx'] for m in segment_metrics]
        seg_l2_errors = [m['L2_error'] for m in segment_metrics]
        seg_linf_errors = [m['Linf_error'] for m in segment_metrics]
        
        ax3.bar(seg_indices, seg_l2_errors, alpha=0.7, label='L2 Error')
        ax3.bar(seg_indices, seg_linf_errors, alpha=0.7, label='L∞ Error')
        ax3.set_xlabel('Segment Index')
        ax3.set_ylabel('Error')
        ax3.set_title('Segment-wise Error Distribution')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Error correlation plot
        ax4 = axes[1, 0]
        ax4.scatter(ref_global, deepoly_global, c='purple', s=4, alpha=0.6)
        # Perfect correlation line
        min_val, max_val = min(ref_global.min(), deepoly_global.min()), max(ref_global.max(), deepoly_global.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Agreement')
        ax4.set_xlabel('Reference Solution')
        ax4.set_ylabel('DeePoly Solution')
        ax4.set_title(f'Solution Correlation (r={metrics["correlation"]:.4f})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Test point distribution by segment
        ax5 = axes[1, 1]
        colors = plt.cm.tab10(np.linspace(0, 1, len(test_data['x_segments_norm'])))
        for seg_idx, x_seg in enumerate(test_data['x_segments_norm']):
            ax5.scatter(x_seg.flatten(), [seg_idx]*len(x_seg), 
                       c=[colors[seg_idx]], s=8, alpha=0.7, label=f'Seg {seg_idx}')
        ax5.set_xlabel('x coordinate')
        ax5.set_ylabel('Segment Index')
        ax5.set_title('Test Point Distribution (Uniform)')
        ax5.grid(True, alpha=0.3)
        if len(test_data['x_segments_norm']) <= 10:
            ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 6. Metrics summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        metrics_text = f"""
Test Data Comparison Metrics:
(Uniform Distribution)

Global Metrics:
• Test Points:     {metrics['n_total_points']}
• Segments:        {metrics['n_segments']}
• L₁ Error:        {metrics['L1_error']:.4e}
• L₂ Error:        {metrics['L2_error']:.4e}
• L∞ Error:        {metrics['Linf_error']:.4e}

Relative Errors:
• Rel. L₁:         {metrics['relative_L1']:.4e}
• Rel. L₂:         {metrics['relative_L2']:.4e}
• Rel. L∞:         {metrics['relative_Linf']:.4e}

Solution Properties:
• Reference Norm:  {metrics['reference_norm']:.4e}
• DeePoly Norm:    {metrics['deepoly_norm']:.4e}
• Correlation:     {metrics['correlation']:.6f}

Time: t = {ref_at_train['time']:.6f}
        """
        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Test data comparison plot saved successfully")
        
    except Exception as e:
        print(f"Warning: Failed to create test data comparison plot: {e}")

if __name__ == "__main__":
    try:
        solver, u_final, model, coeffs = test_single_timestep()
        print("\n✓ Single timestep test completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()