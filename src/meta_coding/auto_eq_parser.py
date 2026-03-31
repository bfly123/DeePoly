from .auto_eq_helpers import generate_all_derivatives
from .auto_eq_linear_parser import (
    build_context,
    collect_derivatives,
    finalize_context,
    parse_linear_equations,
)
from .auto_eq_nonlinear_parser import parse_nonlinear_terms


def parse_equation_to_list_impl(
    equation_str,
    equation_nonlinear_str,
    vars_list,
    spatial_vars,
    const_list,
):
    """Parse PDE equations into linear and nonlinear derivative metadata."""
    context = build_context(vars_list, spatial_vars)
    collect_derivatives(equation_str, vars_list, spatial_vars, context)
    if equation_nonlinear_str:
        collect_derivatives(
            equation_nonlinear_str,
            vars_list,
            spatial_vars,
            context,
            is_nonlinear=True,
        )
    finalize_context(context, vars_list, spatial_vars)
    result = parse_linear_equations(equation_str, vars_list, spatial_vars, const_list, context)
    result_nonlinear = parse_nonlinear_terms(
        equation_nonlinear_str,
        vars_list,
        spatial_vars,
        const_list,
        context,
    )
    all_derivatives = generate_all_derivatives(context["max_deriv_orders"])
    return (
        result,
        context["deriv_orders"],
        context["max_deriv_orders"],
        result_nonlinear,
        all_derivatives,
    )
