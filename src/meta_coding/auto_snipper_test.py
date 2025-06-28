"""
Multi-Operator Equation Parser - Simplified Version
Focus on: 1) Derivatives list 2) Max derivative orders 3) Operator term decomposition
"""

import re
from typing import List, Dict, Any, Optional
from auto_snipper import OperatorParser, parse_operators


def test_basic():
    """Test basic functionality"""
    print("=== Basic Test ===")

    operators = {
        "Linear_Diffusion": ["diff(u,x,2) + diff(u,y,2)", "diff(v,x,2) + diff(v,y,2)"],
        "Nonlinear_Convection": [
            "u*diff(u,x) + v*diff(u,y)",
            "u*diff(v,x) + v*diff(v,y)",
        ],
    }

    constants = {"Re": 100, "Pr": 0.7, "nu": 0.01}
    result = parse_operators(operators, ["u", "v"], ["x", "y"], constants)

    print(f"Derivatives count: {len(result['all_derivatives'])}")
    print(f"Operators count: {len(result['operator_terms'])}")

    print("\nDerivatives list:")
    for i, (name, orders) in enumerate(sorted(result["all_derivatives"].items())):
        print(f"  [{i}] {name}: {orders}")

    print("\nOperator decomposition:")
    for op_name, terms in result["operator_terms"].items():
        print(f"  {op_name}:")
        for i, term in enumerate(terms):
            indices = ", ".join(map(str, term["derivative_indices"]))
            print(f"    [{i}] [{indices}]")


def test_with_constants():
    """Test with physical constants"""
    print("\n=== Constants Test ===")

    operators = {
        "Navier_Stokes": [
            # Momentum equations with Reynolds number
            "u*diff(u,x) + v*diff(u,y) + diff(p,x) - (1/Re)*(diff(u,x,2) + diff(u,y,2))",
            "u*diff(v,x) + v*diff(v,y) + diff(p,y) - (1/Re)*(diff(v,x,2) + diff(v,y,2))",
            "diff(u,x) + diff(v,y)",  # Continuity
        ],
        "Heat_Transfer": [
            # Energy equation with Prandtl number
            "u*diff(T,x) + v*diff(T,y) - (1/(Re*Pr))*(diff(T,x,2) + diff(T,y,2))",
            "alpha*diff(T,x,2) + beta*diff(T,y,2) + gamma*T",
            "k*diff(T,x,2) + rho*cp*u*diff(T,x)",
        ],
        "Reaction_Diffusion": [
            # Chemical reaction with rate constants
            "D1*diff(u,x,2) + D2*diff(u,y,2) + k1*u*v - k2*u**2",
            "D3*diff(v,x,2) + D4*diff(v,y,2) - k1*u*v + k3*v",
            "mu*u + lambda*v + sigma*u*v**2",
        ],
    }

    constants = {
        "Re": 1000,  # Reynolds number
        "Pr": 0.7,  # Prandtl number
        "alpha": 1.5,  # Thermal diffusivity coefficient
        "beta": 2.0,  # Thermal diffusivity coefficient
        "gamma": 0.1,  # Heat source coefficient
        "k": 0.5,  # Thermal conductivity
        "rho": 1.2,  # Density
        "cp": 1005,  # Specific heat
        "D1": 0.01,
        "D2": 0.02,
        "D3": 0.015,
        "D4": 0.025,  # Diffusion coefficients
        "k1": 0.1,
        "k2": 0.05,
        "k3": 0.02,  # Reaction rate constants
        "mu": 0.001,
        "lambda": 0.002,
        "sigma": 0.0001,  # Additional parameters
    }

    variables = ["u", "v", "p", "T"]
    spatial_vars = ["x", "y"]

    result = parse_operators(operators, variables, spatial_vars, constants)

    print(f"Total derivatives found: {len(result['all_derivatives'])}")
    print(f"Total operators: {len(result['operator_terms'])}")

    print("\nAll derivatives list:")
    for i, (name, orders) in enumerate(sorted(result["all_derivatives"].items())):
        var_idx = orders[0]
        spatial_orders = orders[1:]
        print(f"  [{i}] {name}: [var={var_idx}, orders={spatial_orders}]")

    print("\nOperator decomposition with constants:")
    for op_name, terms in result["operator_terms"].items():
        print(f"\n  {op_name}:")
        for i, term in enumerate(terms):
            indices = ", ".join(map(str, term["derivative_indices"]))
            expr = term["symbolic_expr"]
            print(f"    [{i}] Indices: [{indices}]")
            print(f"        Expression: {expr}")


def test_complex_equations():
    """Test complex equations with high-order derivatives, 3D, and mixed terms"""
    print("\n=== Complex Equations Test ===")

    operators = {
        "High_Order_3D_PDE": [
            # 3D high-order PDEs with fourth-order derivatives
            "diff(u,x,4) + diff(u,y,4) + diff(u,z,4) + alpha*diff(u,x,2)*diff(u,y,2)",
            "diff(v,x,3) + diff(v,y,3) + diff(v,z,3) + beta*u*diff(v,x,2)",
            "diff(w,x,2)*diff(w,y,2) + diff(w,z,4) + gamma*u*v*w",
        ],
        "Coupled_Nonlinear_System": [
            # Coupled nonlinear system with mixed derivative terms
            "diff(u,x,2) + diff(u,y,2) + diff(u,z,2) + u*diff(v,x) + v*diff(w,y) + w*diff(u,z)",
            "diff(v,x,3) + u*v*diff(u,x) + w*diff(v,z,2) + sin(u)*cos(v)",
            "diff(w,z,3) + diff(u,x)*diff(v,y)*diff(w,z) + exp(u*v)*w",
        ],
        "Multi_Physics_Problem": [
            # Multi-physics coupling: fluid-structure-thermal coupling
            "rho*(diff(u,t) + u*diff(u,x) + v*diff(u,y) + w*diff(u,z)) + diff(p,x) - mu*(diff(u,x,2) + diff(u,y,2) + diff(u,z,2))",
            "rho*(diff(v,t) + u*diff(v,x) + v*diff(v,y) + w*diff(v,z)) + diff(p,y) - mu*(diff(v,x,2) + diff(v,y,2) + diff(v,z,2))",
            "rho*(diff(w,t) + u*diff(w,x) + v*diff(w,y) + w*diff(w,z)) + diff(p,z) - mu*(diff(w,x,2) + diff(w,y,2) + diff(w,z,2))",
            "diff(u,x) + diff(v,y) + diff(w,z)",  # Continuity equation
            "rho*cp*(diff(T,t) + u*diff(T,x) + v*diff(T,y) + w*diff(T,z)) - k*(diff(T,x,2) + diff(T,y,2) + diff(T,z,2)) - Q",  # Energy equation
        ],
        "Quantum_Mechanics_Analogy": [
            # Schrödinger-like complex equations
            "i*hbar*diff(psi,t) + (hbar**2)/(2*m)*(diff(psi,x,2) + diff(psi,y,2) + diff(psi,z,2)) + V*psi",
            "diff(phi,x,2) + diff(phi,y,2) + diff(phi,z,2) + omega**2*phi + lambda*phi**3",
        ]
    }

    constants = {
        # Material property parameters
        "alpha": 0.01, "beta": 0.02, "gamma": 0.03,
        # Fluid parameters
        "rho": 1000.0,     # Density
        "mu": 0.001,       # Dynamic viscosity
        "nu": 1e-6,        # Kinematic viscosity
        # Thermal properties
        "cp": 4200.0,      # Specific heat capacity
        "k": 0.6,          # Thermal conductivity
        "Q": 1000.0,       # Heat source
        # Quantum mechanics parameters
        "hbar": 1.054e-34, # Reduced Planck constant
        "m": 9.109e-31,    # Electron mass
        "V": 1.0,          # Potential energy
        "omega": 1.0,      # Frequency
        "lambda": 0.1,     # Nonlinear coefficient
        # Mathematical constants
        "i": 1j,           # Imaginary unit
        "pi": 3.14159,
    }

    variables = ["u", "v", "w", "p", "T", "psi", "phi"]  # 7 variables
    spatial_vars = ["x", "y", "z", "t"]  # 3D + time

    try:
        result = parse_operators(operators, variables, spatial_vars, constants)
        
        print(f"✓ Complex equation parsing successful!")
        print(f"✓ Number of variables: {len(variables)}")
        print(f"✓ Spatial dimensions: {len(spatial_vars)}")
        print(f"✓ Unique derivative patterns: {len(result['derivatives'])}")
        print(f"✓ Specific derivative terms: {len(result['all_derivatives'])}")
        print(f"✓ Operator groups: {len(result['operator_terms'])}")

        print("\nUnique derivative patterns (derivatives - variable-independent):")
        for i, pattern in enumerate(result["derivatives"][:15]):  # Show only first 15
            description = []
            for j, order in enumerate(pattern):
                if order > 0:
                    description.append(f"{spatial_vars[j]}^{order}")
            if not description:
                description = ["0th order"]
            print(f"  [{i}] {pattern} -> {', '.join(description)}")
        if len(result["derivatives"]) > 15:
            print(f"  ... {len(result['derivatives'])-15} more patterns")

        print(f"\nHigh-order derivative statistics:")
        max_orders = {}
        for pattern in result["derivatives"]:
            for dim_idx, order in enumerate(pattern):
                dim_name = spatial_vars[dim_idx]
                max_orders[dim_name] = max(max_orders.get(dim_name, 0), order)
        
        for dim, max_order in max_orders.items():
            print(f"  Maximum {dim}-direction derivative order: {max_order}")

        print(f"\nComplex derivative terms examples:")
        complex_terms = []
        for name, info in result["all_derivatives"].items():
            var_idx, deriv_idx = info
            pattern = result["derivatives"][deriv_idx]
            max_order = max(pattern)
            if max_order >= 3:  # Show 3rd order and above
                complex_terms.append((name, var_idx, deriv_idx, pattern, max_order))
        
        # Sort by highest order
        complex_terms.sort(key=lambda x: x[4], reverse=True)
        
        for name, var_idx, deriv_idx, pattern, max_order in complex_terms[:10]:
            var_name = variables[var_idx]
            print(f"  {name}: var={var_name}(idx={var_idx}), pattern={pattern}, max_order={max_order}")

        return True

    except Exception as e:
        print(f"✗ Complex equation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extreme_complexity():
    """Test extremely complex equations with mixed derivatives and special cases"""
    print("\n=== Extreme Complexity Test ===")

    operators = {
        "Mixed_Derivatives_PDE": [
            # Note: Current parser may not support mixed derivatives, but we can test other complex cases
            "diff(u,x,6) + diff(u,y,6) + diff(u,z,6) + A*u*diff(u,x,5)",
            "diff(v,x,5)*diff(v,y) + diff(v,z,7) + B*exp(u)*sin(v)",
            "diff(w,t,2) + diff(w,x,8) + C*u*v*w*diff(w,x,3)",
        ],
        "Ultra_High_Order": [
            # Ultra-high-order derivative equations
            "diff(psi,x,10) + diff(psi,y,10) + diff(psi,z,10) + omega*psi",
            "diff(phi,x,12) + diff(phi,t,3) + lambda*phi**5",
        ],
        "Many_Variables_System": [
            # Multi-variable coupled system
            "alpha1*u1 + beta1*diff(u2,x) + gamma1*diff(u3,y) + delta1*diff(u4,z)",
            "alpha2*diff(u1,x,2) + beta2*u2 + gamma2*diff(u3,x) + delta2*diff(u4,y)",
            "alpha3*diff(u1,y) + beta3*diff(u2,z) + gamma3*u3 + delta3*diff(u4,x,2)",
            "alpha4*diff(u1,z,3) + beta4*diff(u2,y,2) + gamma4*diff(u3,x,3) + delta4*u4",
        ],
        "Fractal_Like_Equation": [
            # Fractal-like equation (highly nonlinear)
            "diff(f,x,7) + diff(f,y,7) + A*f*diff(f,x,3)*diff(f,y,2) + B*f**3",
            "diff(g,t,4) + diff(g,x,9) + C*sin(f)*cos(g)*diff(g,x,4)",
        ]
    }

    constants = {
        "A": 1.5, "B": 2.3, "C": 3.7,
        "alpha1": 0.1, "alpha2": 0.2, "alpha3": 0.3, "alpha4": 0.4,
        "beta1": 0.5, "beta2": 0.6, "beta3": 0.7, "beta4": 0.8,
        "gamma1": 0.9, "gamma2": 1.1, "gamma3": 1.2, "gamma4": 1.3,
        "delta1": 1.4, "delta2": 1.5, "delta3": 1.6, "delta4": 1.7,
        "omega": 2.5, "lambda": 3.8,
    }

    variables = ["u", "v", "w", "psi", "phi", "u1", "u2", "u3", "u4", "f", "g"]  # 11 variables
    spatial_vars = ["x", "y", "z", "t"]  # 4D

    try:
        result = parse_operators(operators, variables, spatial_vars, constants)
        
        print(f"🚀 Extreme complex equation parsing successful!")
        print(f"📊 Statistics:")
        print(f"   - Number of variables: {len(variables)}")
        print(f"   - Spatial dimensions: {len(spatial_vars)}")
        print(f"   - Unique derivative patterns: {len(result['derivatives'])}")
        print(f"   - Specific derivative terms: {len(result['all_derivatives'])}")
        print(f"   - Operator groups: {len(result['operator_terms'])}")

        # Analyze highest-order derivatives
        max_total_order = 0
        highest_order_terms = []
        
        for name, info in result["all_derivatives"].items():
            var_idx, deriv_idx = info
            pattern = result["derivatives"][deriv_idx]
            total_order = sum(pattern)
            max_order_in_term = max(pattern)
            
            if max_order_in_term >= 7:  # 7th order and above
                highest_order_terms.append((name, variables[var_idx], pattern, max_order_in_term, total_order))
            
            max_total_order = max(max_total_order, total_order)

        print(f"\n🎯 Ultra-high-order derivative analysis:")
        print(f"   - Highest single-direction derivative order in system: {max([max(p) for p in result['derivatives']])}")
        print(f"   - Highest total derivative order in system: {max_total_order}")

        # Show ultra-high-order derivative terms
        if highest_order_terms:
            print(f"\n🔥 Ultra-high-order derivative terms (7th order and above):")
            highest_order_terms.sort(key=lambda x: x[3], reverse=True)
            for name, var_name, pattern, max_order, total_order in highest_order_terms[:15]:
                print(f"   {name}: var={var_name}, pattern={pattern}, max_order={max_order}, total_order={total_order}")

        # Statistics of derivative pattern distribution
        print(f"\n📈 Derivative order distribution:")
        order_distribution = {}
        for pattern in result["derivatives"]:
            max_order = max(pattern)
            order_distribution[max_order] = order_distribution.get(max_order, 0) + 1

        for order in sorted(order_distribution.keys()):
            count = order_distribution[order]
            print(f"   {order}th order derivative patterns: {count} patterns")

        return True

    except Exception as e:
        print(f"💥 Extreme complex equation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_math_functions():
    """Test mathematical functions (sin, cos, exp, log, etc.)"""
    print("\n=== Mathematical Functions Test ===")

    operators = {
        "Trigonometric_Terms": [
            # Trigonometric functions with derivatives and constants
            "A*sin(omega*u)*diff(u,x) + B*cos(omega*v)*diff(v,y)",
            "C*tan(u)*diff(u,x,2) + D*sin(u)*cos(v)*diff(u,y)",
            "E*sin(u*v) + F*cos(diff(u,x)) + G*tan(u**2)",
        ],
        "Exponential_Terms": [
            # Exponential and logarithmic functions with constants
            "alpha*exp(beta*u)*diff(u,x) + gamma*log(delta*v)*diff(v,y)",
            "a*exp(b*u*v)*diff(u,x,2) + c*ln(d*u)*diff(v,x)*diff(v,y)",
            "K1*exp(sin(u)) + K2*log(cos(v)) + K3*exp(diff(u,x))",
        ],
        "Mixed_Complex": [
            # Complex combinations with constants
            "A*sin(u)*exp(B*v)*diff(u,x) + C*cos(u)*log(D*v)*diff(v,y)",
            "E*exp(sin(F*u))*diff(u,x,2) + G*log(cos(H*v))*diff(v,x)*diff(v,y)",
            "I*sqrt(sin(J*u)**2 + cos(K*v)**2)*diff(u,x)*diff(v,y)",
        ],
    }

    constants = {
        "A": 1.0,
        "B": 2.0,
        "C": 3.0,
        "D": 4.0,
        "E": 5.0,
        "F": 6.0,
        "G": 7.0,
        "H": 8.0,
        "I": 9.0,
        "J": 10.0,
        "K": 11.0,
        "omega": 3.14159,
        "alpha": 0.1,
        "beta": 0.2,
        "gamma": 0.3,
        "delta": 0.4,
        "a": 1.1,
        "b": 1.2,
        "c": 1.3,
        "d": 1.4,
        "K1": 2.1,
        "K2": 2.2,
        "K3": 2.3,
    }

    variables = ["u", "v", "w"]
    spatial_vars = ["x", "y"]

    result = parse_operators(operators, variables, spatial_vars, constants)

    print(f"Total derivatives found: {len(result['all_derivatives'])}")
    print(f"Total operators: {len(result['operator_terms'])}")

    print("\nOperator decomposition with mathematical functions and constants:")
    for op_name, terms in result["operator_terms"].items():
        print(f"\n  {op_name}:")
        for i, term in enumerate(terms):
            indices = ", ".join(map(str, term["derivative_indices"]))
            expr = term["symbolic_expr"]
            print(f"    [{i}] Indices: [{indices}]")
            print(f"        Expression: {expr}")


if __name__ == "__main__":
    test_basic()
    test_with_constants()
    test_complex_equations()  # New complex equation tests
    test_extreme_complexity()  # New extreme complexity tests
    test_math_functions()
    print("\n=== All Tests Completed ===")
