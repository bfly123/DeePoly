import numpy as np
import torch
from math import pi, sin

def generate_reference_solution(x):
    """
    Generate reference solution for linear convection equation
    with piecewise initial condition defined in config.json
    
    Args:
        x: Input coordinates (n_points, 2) where first column is t and second is x
    Returns:
        u: Solution values (n_points, 1)
    """
    t_coords = x[:, 0]
    x_coords = x[:, 1]
    
    u = np.zeros_like(x_coords)
    
    # For linear convection equation: ∂u/∂t + 0.3*∂u/∂x = 0
    # Solution along characteristics: u(x,t) = u_0(x - 0.3*t)
    # where u_0 is the initial condition at t=0
    
    for i in range(len(x_coords)):
        x_val = x_coords[i]
        t_val = t_coords[i]
        
        # Calculate the characteristic coordinate (where this value originated at t=0)
        x_char = x_val - 0.3 * t_val
        
            # Use the piecewise initial condition from config.json at t=0
        u[i] =  np.tanh(100*(x_char-0.3))
    return u.reshape(-1, 1)

def get_initial_condition(x_val):
    """
    Evaluate the smooth initial condition exactly as defined in config.json
    
    Args:
        x_val: x coordinate value
    Returns:
        u_val: Initial condition value at x_val
    """
    # Piecewise function from config.json:
    # Range [0, 0.7]: smooth function with zero boundary conditions
    if 0 <= x_val <= 0.7:
        return 16 * (x_val/0.7)**2 * (1 - x_val/0.7)**2
    
    # Range [0.7, 1.0]: expression = "0" 
    elif 0.7 < x_val <= 1.0:
        return 0
    
    # Outside domain
    else:
        return 0

def generate_global_field(x):
    """
    Generate global field for linear convection equation
    
    Args:
        x: Input coordinates (n_points, 2)
    Returns:
        u: Field values (n_points, 1)
    """
    return generate_reference_solution(x)

def exact_solution(x):
    """
    Alias for generate_reference_solution for compatibility
    
    Args:
        x: Input coordinates (n_points, 2)
    Returns:
        u: Solution values (n_points, 1)  
    """
    return generate_reference_solution(x)

def generate_source_term(x):
    """
    Generate source term for the equation (should be 0 for pure convection)
    
    Args:
        x: Input coordinates (n_points, 2)
    Returns:
        source: Source term values (n_points, 1)
    """
    return np.zeros((x.shape[0], 1))

def plot_initial_condition():
    """
    Utility function to visualize the initial condition
    """
    import matplotlib.pyplot as plt
    
    x_vals = np.linspace(0, 1, 1000)
    u_vals = [get_initial_condition(x) for x in x_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, u_vals, 'b-', linewidth=2, label='Initial Condition u(x,0)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Piecewise Initial Condition for Linear Convection')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Mark the piecewise boundaries
    plt.axvline(x=0.1, color='r', linestyle='--', alpha=0.7, label='x=0.1')
    plt.axvline(x=0.6, color='r', linestyle='--', alpha=0.7, label='x=0.6')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_solution_evolution():
    """
    Utility function to visualize solution evolution over time
    """
    import matplotlib.pyplot as plt
    
    x_vals = np.linspace(0, 1, 200)
    time_vals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    plt.figure(figsize=(12, 8))
    
    for t in time_vals:
        u_vals = []
        for x in x_vals:
            coords = np.array([[t, x]])
            u_val = generate_reference_solution(coords)[0, 0]
            u_vals.append(u_val)
        
        plt.plot(x_vals, u_vals, label=f't = {t}', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Linear Convection Solution Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the implementation
    print("Testing piecewise initial condition...")
    
    # Test specific points
    test_points = [0.05, 0.15, 0.35, 0.45, 0.7]
    for x_test in test_points:
        u_val = get_initial_condition(x_test)
        print(f"u({x_test}, 0) = {u_val:.6f}")
    
    # Uncomment to visualize
    # plot_initial_condition()
    # plot_solution_evolution()