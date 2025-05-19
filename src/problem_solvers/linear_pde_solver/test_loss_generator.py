import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.problem_solvers.linear_pde_solver.auto_replace_loss import update_physics_loss_code

def test_simple_equation():
    """Test a simple equation: ∂u/∂x + ∂u/∂y = 0"""
    # Define dimensions and variables
    dimensions = ["x", "y"]
    vars_list = ["u"]
    
    # Define the simple equation
    equations = [
        "diff(u,x) + diff(u,y)"  # Simple first-order equation
    ]
    
    # Update the physics loss code
    update_physics_loss_code(
        linear_equations=equations,
        vars_list=vars_list,
        spatial_vars=dimensions,
        model_path="src/problem_solvers/linear_pde_solver/core/net.py"
    )

def test_laplace_equation():
    """Test Laplace equation: ∇²u = 0"""
    # Define dimensions and variables
    dimensions = ["x", "y"]
    vars_list = ["u"]
    
    # Define the Laplace equation
    equations = [
        "diff(u,x,2) + diff(u,y,2)"  # Laplace equation
    ]
    
    # Update the physics loss code
    update_physics_loss_code(
        linear_equations=equations,
        vars_list=vars_list,
        spatial_vars=dimensions,
        model_path="src/problem_solvers/linear_pde_solver/core/net.py"
    )

def test_heat_equation():
    """Test heat equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)"""
    # Define dimensions and variables
    dimensions = ["x", "y", "t"]
    vars_list = ["u"]
    const_list = [{"alpha": 1.0}]  # Thermal diffusivity
    
    # Define the heat equation
    equations = [
        "diff(u,t) - alpha*(diff(u,x,2) + diff(u,y,2))"  # Heat equation
    ]
    
    # Update the physics loss code
    update_physics_loss_code(
        linear_equations=equations,
        vars_list=vars_list,
        spatial_vars=dimensions,
        const_list=const_list,
        model_path="src/problem_solvers/linear_pde_solver/core/net.py"
    )

def test_navier_stokes_2d():
    """Test 2D Navier-Stokes equations"""
    # Define dimensions and variables
    dimensions = ["x", "y"]
    vars_list = ["u", "v", "p"]
    const_list = [{"Re": 100}]  # Reynolds number
    
    # Define the Navier-Stokes equations
    linear_equations = [
        "diff(u,x) + diff(v,y)",  # Continuity equation
        "diff(p,x) - (diff(u,x,2) + diff(u,y,2))/Re",  # x-momentum (linear terms)
        "diff(p,y) - (diff(v,x,2) + diff(v,y,2))/Re"   # y-momentum (linear terms)
    ]
    
    nonlinear_equations = [
        "0",  # No nonlinear terms in continuity
        "u*diff(u,x) + v*diff(u,y)",  # x-momentum (nonlinear terms)
        "u*diff(v,x) + v*diff(v,y)"   # y-momentum (nonlinear terms)
    ]
    
    # Update the physics loss code
    update_physics_loss_code(
        linear_equations=linear_equations,
        nonlinear_equations=nonlinear_equations,
        vars_list=vars_list,
        spatial_vars=dimensions,
        const_list=const_list,
        model_path="src/problem_solvers/linear_pde_solver/core/net.py"
    )

def test_wave_equation():
    """Test wave equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)"""
    # Define dimensions and variables
    dimensions = ["x", "y", "t"]
    vars_list = ["u"]
    const_list = [{"c": 1.0}]  # Wave speed
    
    # Define the wave equation
    equations = [
        "diff(u,t,2) - c*c*(diff(u,x,2) + diff(u,y,2))"  # Wave equation
    ]
    
    # Update the physics loss code
    update_physics_loss_code(
        linear_equations=equations,
        vars_list=vars_list,
        spatial_vars=dimensions,
        const_list=const_list,
        model_path="src/problem_solvers/linear_pde_solver/core/net.py"
    )

if __name__ == "__main__":
    print("Testing different PDE systems...")
    
    print("\n1. Testing simple equation...")
    test_simple_equation()
    
    print("\n2. Testing Laplace equation...")
    test_laplace_equation()
    
    print("\n3. Testing Heat equation...")
    test_heat_equation()
    
    print("\n4. Testing 2D Navier-Stokes equations...")
    test_navier_stokes_2d()
    
    print("\n5. Testing Wave equation...")
    test_wave_equation()
    
    print("\nAll tests completed!") 