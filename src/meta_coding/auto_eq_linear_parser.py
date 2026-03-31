from .auto_eq_helpers import (
    extract_diff_content,
    get_deriv_name,
    parse_coefficient,
    parse_diff_term,
)


def build_context(vars_list, spatial_vars):
    n_dim = len(spatial_vars)
    return {
        "n_dim": n_dim,
        "max_deriv_orders": [[0] * n_dim for _ in range(len(vars_list))],
        "deriv_orders_set": {tuple([0] * n_dim)},
    }


def collect_derivatives(equations, vars_list, spatial_vars, context, is_nonlinear=False):
    for equation in equations:
        if not equation or equation == "0":
            continue
        for term in split_terms(equation):
            if is_nonlinear and "*" in term:
                collect_term_derivatives(term.split("*"), vars_list, spatial_vars, context)
                continue
            collect_term_derivatives([term], vars_list, spatial_vars, context)


def collect_term_derivatives(terms, vars_list, spatial_vars, context):
    for term in terms:
        if "diff(" not in term:
            continue
        content = extract_diff_content(term)
        if not content:
            continue
        var_name, deriv_var, order = parse_diff_term(content, vars_list, spatial_vars)
        if not var_name:
            continue
        update_max_derivative(var_name, deriv_var, order, vars_list, spatial_vars, context)


def update_max_derivative(var_name, deriv_var, order, vars_list, spatial_vars, context):
    var_index = vars_list.index(var_name)
    dim_index = spatial_vars.index(deriv_var)
    context["max_deriv_orders"][var_index][dim_index] = max(
        context["max_deriv_orders"][var_index][dim_index],
        order,
    )
    deriv_tuple = [0] * context["n_dim"]
    deriv_tuple[dim_index] = order
    context["deriv_orders_set"].add(tuple(deriv_tuple))


def finalize_context(context, vars_list, spatial_vars):
    deriv_orders = sorted(list(context["deriv_orders_set"]))
    deriv_to_index = {deriv: idx for idx, deriv in enumerate(deriv_orders)}
    deriv_names = {}
    for deriv_idx, deriv_tuple in enumerate(deriv_orders):
        for var_idx, var in enumerate(vars_list):
            deriv_names[f"{var_idx},{deriv_idx}"] = get_deriv_name(var, deriv_tuple, spatial_vars)
    context["deriv_orders"] = deriv_orders
    context["deriv_to_index"] = deriv_to_index
    context["deriv_names"] = deriv_names


def parse_linear_equations(equations, vars_list, spatial_vars, const_list, context):
    result = []
    for equation in equations:
        if not equation or equation == "0":
            result.append([])
            continue
        result.append(
            parse_single_linear_equation(
                equation,
                vars_list,
                spatial_vars,
                const_list,
                context,
            )
        )
    return result


def parse_single_linear_equation(equation, vars_list, spatial_vars, const_list, context):
    eq_terms = []
    for term in split_terms(equation):
        parsed_term = parse_linear_term(term, vars_list, spatial_vars, const_list, context)
        if parsed_term is not None:
            eq_terms.append(parsed_term)
    return eq_terms


def parse_linear_term(term, vars_list, spatial_vars, const_list, context):
    coef, num_part = split_term_sign_and_division(term, const_list)
    if coef is None:
        return None
    if "diff(" in num_part:
        return parse_diff_linear_term(
            coef,
            num_part,
            vars_list,
            spatial_vars,
            const_list,
            context,
        )
    variable_term = parse_plain_variable_term(num_part, vars_list, const_list, context)
    if variable_term is not None:
        variable_term[0] *= coef
        return variable_term
    const_val = parse_coefficient(num_part, const_list)
    if const_val is None:
        return None
    return [coef * const_val, -1, -1, "const"]


def split_term_sign_and_division(term, const_list):
    coef = -1.0 if term.startswith("-") else 1.0
    if term.startswith("-"):
        term = term[1:]
    term_parts = term.split("/")
    num_part = term_parts[0]
    if len(term_parts) == 1:
        return coef, num_part
    denom_val = resolve_denominator(term_parts[1], const_list)
    if denom_val is None:
        print(f"警告: 无法解析分母: {term_parts[1]}")
        return None, None
    return coef / denom_val, num_part


def resolve_denominator(denom, const_list):
    for const_dict in const_list:
        if denom in const_dict:
            return const_dict[denom]
    try:
        return float(denom)
    except ValueError:
        return None


def parse_diff_linear_term(coef, num_part, vars_list, spatial_vars, const_list, context):
    diff_idx = num_part.find("diff(")
    coef_val = 1.0
    if diff_idx > 0:
        coef_val = parse_coefficient(num_part[:diff_idx], const_list)
    content = extract_diff_content(num_part[diff_idx:])
    if not content:
        return None
    var_name, deriv_var, deriv_order = parse_diff_term(content, vars_list, spatial_vars)
    if not var_name:
        return None
    var_index = vars_list.index(var_name)
    dim_index = spatial_vars.index(deriv_var)
    deriv_tuple = [0] * context["n_dim"]
    deriv_tuple[dim_index] = deriv_order
    deriv_index = context["deriv_to_index"][tuple(deriv_tuple)]
    term_name = context["deriv_names"][f"{var_index},{deriv_index}"]
    return [coef * coef_val, var_index, deriv_index, term_name]


def parse_plain_variable_term(num_part, vars_list, const_list, context):
    if num_part in vars_list:
        return build_zero_derivative_term(1.0, num_part, vars_list, context)
    if "*" not in num_part:
        return None

    factors = [factor for factor in num_part.split("*") if factor]
    variable_factors = [factor for factor in factors if factor in vars_list]
    if len(variable_factors) != 1:
        return None

    variable_name = variable_factors[0]
    coefficient = 1.0
    for factor in factors:
        if factor == variable_name:
            continue
        coefficient *= parse_coefficient(factor, const_list)
    return build_zero_derivative_term(coefficient, variable_name, vars_list, context)


def build_zero_derivative_term(coefficient, variable_name, vars_list, context):
    var_index = vars_list.index(variable_name)
    deriv_index = context["deriv_to_index"][tuple([0] * context["n_dim"])]
    term_name = context["deriv_names"][f"{var_index},{deriv_index}"]
    return [coefficient, var_index, deriv_index, term_name]


def split_terms(equation):
    equation = equation.replace("-", "+-").replace(" ", "")
    if equation.startswith("+"):
        equation = equation[1:]
    return [term for term in equation.split("+") if term]
