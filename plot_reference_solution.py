#!/usr/bin/env python3
"""
Plot reference solution at T=0.1 for Allen-Cahn equation
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from ac_reference_solver import ACReferenceSolver

def plot_reference_solution():
    """Plot reference solution at T=0.1"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize reference solver
    ref_solver = ACReferenceSolver(current_dir)
    
    # Get reference solution at T=0.1
    T = 0.1
    ref_data = ref_solver.solve_reference(T=T, dt=0.1, N_points=512)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Reference solution at T=0.1
    plt.subplot(2, 2, 1)
    plt.plot(ref_data['x'], ref_data['u'], 'b-', linewidth=2, label=f'Reference at t={T}')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Allen-Cahn Reference Solution at T={T}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Initial condition comparison
    plt.subplot(2, 2, 2)
    x_init = ref_data['x']
    u_init = ref_solver.get_initial_condition(x_init)
    plt.plot(x_init, u_init, 'r--', linewidth=2, label='Initial: $x^2\\cos(\\pi x)$')
    plt.plot(ref_data['x'], ref_data['u'], 'b-', linewidth=2, label=f'Reference at t={T}')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Initial vs Reference Solution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Zoom in on central region
    plt.subplot(2, 2, 3)
    mask = (ref_data['x'] >= -0.5) & (ref_data['x'] <= 0.5)
    plt.plot(ref_data['x'][mask], ref_data['u'][mask], 'b-', linewidth=2, marker='o', markersize=3)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Reference Solution (Central Region) at T={T}')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Solution statistics
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.9, f'Reference Solution Statistics at T={T:.3f}:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'Min value: {np.min(ref_data["u"]):.6f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'Max value: {np.max(ref_data["u"]):.6f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'L2 norm: {np.linalg.norm(ref_data["u"]):.6f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'Mean: {np.mean(ref_data["u"]):.6f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'Std: {np.std(ref_data["u"]):.6f}', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f'Domain: [{ref_data["x"].min():.1f}, {ref_data["x"].max():.1f}]', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'Points: {len(ref_data["x"])}', fontsize=10, transform=plt.gca().transAxes)
    
    # Initial condition statistics
    plt.text(0.1, 0.05, f'Initial condition L2 norm: {np.linalg.norm(u_init):.6f}', fontsize=10, transform=plt.gca().transAxes)
    
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(current_dir, "cases", "Time_pde_cases", "AC_equation", "results", "reference_solution_T_0.1.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Reference solution plot saved to: {output_path}")
    
    # Also show some key points
    print(f"\n=== Reference Solution at T={T} ===")
    print(f"Domain: [{ref_data['x'].min():.6f}, {ref_data['x'].max():.6f}]")
    print(f"Solution range: [{np.min(ref_data['u']):.6f}, {np.max(ref_data['u']):.6f}]")
    print(f"L2 norm: {np.linalg.norm(ref_data['u']):.6f}")
    print(f"Number of points: {len(ref_data['x'])}")
    print(f"Actual time from reference: {ref_data['time']:.6f}")
    
    # Show some sample values
    print(f"\nSample values:")
    indices = [0, 128, 256, 384, 511]  # Sample at different positions
    for i in indices:
        if i < len(ref_data['x']):
            print(f"  x={ref_data['x'][i]:6.3f}, u={ref_data['u'][i]:8.6f}")
    
    plt.show()

if __name__ == "__main__":
    plot_reference_solution()