from .auto_eq_helpers import extract_diff_content, parse_diff_term


def parse_nonlinear_terms(equations, vars_list, spatial_vars, const_list, context):
    nonlinear_terms_set = set()
    if not equations:
        return []
    for equation in equations:
        if not equation or equation == "0":
            continue
        for term in split_terms(equation):
            if "*" not in term:
                continue
            factors = parse_nonlinear_factors(
                term,
                vars_list,
                spatial_vars,
                const_list,
                context,
            )
            if len(factors) >= 2:
                nonlinear_terms_set.update(factors)

    result_nonlinear = []
    for var_idx, deriv_idx in sorted(nonlinear_terms_set):
        term_name = context["deriv_names"][f"{var_idx},{deriv_idx}"]
        result_nonlinear.append([var_idx, deriv_idx, term_name])
    return result_nonlinear


def parse_nonlinear_factors(term, vars_list, spatial_vars, const_list, context):
    factors = []
    for sub_term in term.split("*"):
        sub_term = sub_term.strip()
        if is_numeric(sub_term) or is_constant(sub_term, const_list):
            continue
        parsed_factor = parse_nonlinear_factor(
            sub_term,
            vars_list,
            spatial_vars,
            context,
        )
        if parsed_factor is not None:
            factors.append(parsed_factor)
    return factors


def parse_nonlinear_factor(sub_term, vars_list, spatial_vars, context):
    if "diff(" in sub_term:
        content = extract_diff_content(sub_term)
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
        return (var_index, deriv_index)
    if sub_term in vars_list:
        var_index = vars_list.index(sub_term)
        deriv_index = context["deriv_to_index"][tuple([0] * context["n_dim"])]
        return (var_index, deriv_index)
    return None


def split_terms(equation):
    equation = equation.replace("-", "+-").replace(" ", "")
    if equation.startswith("+"):
        equation = equation[1:]
    return [term for term in equation.split("+") if term]


def is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_constant(value: str, const_list) -> bool:
    return any(value in const_dict for const_dict in const_list)
