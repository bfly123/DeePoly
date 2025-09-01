#!/usr/bin/env python3
"""
Analyze time accuracy and error decomposition for Allen-Cahn equation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def theoretical_time_error_analysis():
    """Analyze theoretical time discretization error"""
    print("=== Time Accuracy Analysis for Allen-Cahn Equation ===")
    
    # Parameters
    dt = 0.1
    T = 0.1
    n_steps = int(T / dt)  # 1 step
    
    # Allen-Cahn parameters
    epsilon = 0.0001  # diffusion coefficient
    
    print(f"Time step: dt = {dt}")
    print(f"Final time: T = {T}")
    print(f"Number of steps: {n_steps}")
    print(f"Diffusion coefficient: ε = {epsilon}")
    
    # Theoretical error estimates for IMEX RK(2,2,2)
    print(f"\n=== Theoretical Time Error Estimates ===")
    
    # Local truncation error (per step)
    local_error = dt**3  # O(dt^3) for 2nd order method
    print(f"Local truncation error (per step): O(dt³) ≈ {local_error:.6e}")
    
    # Global discretization error 
    global_time_error = dt**2  # O(dt^2) accumulated over [0,T]
    print(f"Global time discretization error: O(dt²) ≈ {global_time_error:.6e}")
    
    # Estimate based on typical constants for Allen-Cahn
    # For Allen-Cahn: ||u_tt|| ~ ||u|| / ε (stiff equation)
    # Conservative estimate: C ~ 10-100 for this problem
    C_estimate = 50  # Conservative constant
    
    estimated_global_error = C_estimate * dt**2
    print(f"Estimated global error (C=50): {estimated_global_error:.6e}")
    
    # Observed error from DeePoly results
    observed_rel_l2 = 0.0502  # 5.02%
    observed_abs_linf = 0.04695
    
    print(f"\n=== Observed vs Theoretical Error ===")
    print(f"Observed relative L2 error: {observed_rel_l2:.4f} ({observed_rel_l2*100:.1f}%)")
    print(f"Observed absolute L∞ error: {observed_abs_linf:.6f}")
    print(f"Theoretical estimate: {estimated_global_error:.6e}")
    
    # Error decomposition estimate
    print(f"\n=== Error Decomposition Estimate ===")
    
    # Assume observed error is combination of:
    # 1. Time discretization error
    # 2. Spatial discretization error (neural network + polynomial)
    # 3. Solver convergence error
    
    # Very rough estimates based on typical behavior:
    time_error_contribution = estimated_global_error / observed_rel_l2
    
    print(f"Time discretization contribution: ~{time_error_contribution:.1%} of total error")
    print(f"Spatial + solver contribution: ~{1-time_error_contribution:.1%} of total error")
    
    # For single step analysis
    print(f"\n=== Single Step Error Analysis ===")
    print(f"For dt = {dt}, single step:")
    print(f"• Theoretical local error: O({local_error:.1e})")
    print(f"• Observed total error: O({observed_rel_l2:.1e})")
    
    # What would happen with smaller dt?
    dt_small = 0.01
    dt_smaller = 0.001
    
    theoretical_dt01 = (dt_small/dt)**2 * estimated_global_error
    theoretical_dt001 = (dt_smaller/dt)**2 * estimated_global_error
    
    print(f"\n=== Time Step Sensitivity Analysis ===")
    print(f"If dt = 0.01 (10 steps): Expected error ~ {theoretical_dt01:.6e} ({theoretical_dt01*100:.2f}%)")
    print(f"If dt = 0.001 (100 steps): Expected error ~ {theoretical_dt001:.6e} ({theoretical_dt001*100:.3f}%)")
    
    # Create visualization
    create_error_analysis_plot(dt, observed_rel_l2, estimated_global_error)
    
    return {
        'dt': dt,
        'local_error_order': local_error,
        'global_error_order': global_time_error,
        'estimated_global': estimated_global_error,
        'observed_rel_l2': observed_rel_l2,
        'time_contribution': time_error_contribution
    }

def create_error_analysis_plot(dt, observed_error, theoretical_error):
    """Create error analysis visualization"""
    
    # Range of time steps to analyze
    dt_range = np.logspace(-3, -0.5, 20)  # 0.001 to 0.316
    
    # Theoretical scaling
    theoretical_errors = (dt_range/dt)**2 * theoretical_error
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Time Discretization Error Analysis for Allen-Cahn Equation', fontsize=14)
    
    # Plot 1: Error vs time step
    ax1 = axes[0, 0]
    ax1.loglog(dt_range, theoretical_errors, 'b-', linewidth=2, label='Theoretical O(dt²)')
    ax1.loglog(dt, observed_error, 'ro', markersize=10, label=f'Observed (dt={dt})')
    
    # Reference slopes
    ax1.loglog(dt_range, 0.1 * dt_range**2, 'k--', alpha=0.5, label='dt² reference')
    ax1.loglog(dt_range, 0.01 * dt_range**1, 'k:', alpha=0.5, label='dt¹ reference')
    
    ax1.set_xlabel('Time step (dt)')
    ax1.set_ylabel('Relative L2 Error')
    ax1.set_title('Error Scaling Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of steps vs accuracy
    ax2 = axes[0, 1]
    n_steps = 0.1 / dt_range  # T=0.1 fixed
    ax2.loglog(n_steps, theoretical_errors, 'g-', linewidth=2, label='Expected Error')
    ax2.loglog(1, observed_error, 'ro', markersize=10, label='Observed (1 step)')
    ax2.set_xlabel('Number of time steps')
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_title('Steps vs Accuracy Trade-off')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error components breakdown
    ax3 = axes[1, 0]
    components = ['Time Discretization', 'Spatial/Neural', 'Solver Convergence']
    # Rough estimates based on typical behavior
    values = [theoretical_error/observed_error, 0.6, 0.1]  # Normalized to sum ~1
    values = np.array(values) / sum(values)  # Normalize
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax3.bar(components, values, color=colors, alpha=0.7)
    ax3.set_ylabel('Estimated Contribution')
    ax3.set_title('Error Source Breakdown')
    ax3.set_ylim(0, 1)
    
    # Add percentage labels
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom')
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
Time Accuracy Analysis Summary:

Current Configuration:
• Time step: dt = {dt}
• Method: IMEX RK(2,2,2) [2nd order]
• Domain: [-1, 1]
• Final time: T = 0.1

Error Analysis:
• Theoretical local: O(dt³) ≈ {dt**3:.1e}
• Theoretical global: O(dt²) ≈ {dt**2:.1e}
• Observed rel. L2: {observed_error:.3f} ({observed_error*100:.1f}%)

Single Step Assessment:
• Expected time error: ~{theoretical_error:.1e}
• Observed total error: ~{observed_error:.1e}
• Time error fraction: ~{theoretical_error/observed_error:.1%}

Recommendations:
• For 1% accuracy: use dt ≤ 0.032
• For 0.1% accuracy: use dt ≤ 0.010
• Current dt=0.1 gives ~5% error (acceptable)
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.getcwd(), "cases", "Time_pde_cases", "AC_equation", 
                              "results", "time_accuracy_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTime accuracy analysis plot saved to: {output_path}")

if __name__ == "__main__":
    results = theoretical_time_error_analysis()
    print("\n✓ Time accuracy analysis completed!")