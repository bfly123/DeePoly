from itertools import product


def split_equations(equation_str):
    """Split an equation string while respecting nested parentheses."""
    equations = []
    current_eq = ""
    paren_count = 0
    for char in equation_str:
        if char == "(":
            paren_count += 1
        elif char == ")":
            paren_count -= 1
        elif char == "," and paren_count == 0:
            equations.append(current_eq.strip())
            current_eq = ""
            continue
        current_eq += char
    if current_eq:
        equations.append(current_eq.strip())
    return equations


def extract_diff_content(term):
    """Extract the content of a diff(...) expression."""
    if "diff(" not in term:
        return None
    start_idx = term.index("diff(") + 5
    paren_count = 1
    end_idx = start_idx
    for index in range(start_idx, len(term)):
        if term[index] == "(":
            paren_count += 1
        elif term[index] == ")":
            paren_count -= 1
            if paren_count == 0:
                end_idx = index
                break
    return term[start_idx:end_idx]


def parse_diff_term(content, vars_list, spatial_vars):
    """Parse a diff(...) payload into variable, dimension, and order."""
    parts = split_equations(content)
    if len(parts) < 2:
        return None, None, 0
    var_name = parts[0].strip()
    deriv_var = parts[1].strip()
    if var_name not in vars_list or deriv_var not in spatial_vars:
        return None, None, 0
    deriv_order = 1
    if len(parts) > 2:
        try:
            deriv_order = int(parts[2].strip())
        except ValueError:
            return None, None, 0
    return var_name, deriv_var, deriv_order


def parse_coefficient(term_part, const_list):
    """Parse a numeric or constant coefficient fragment."""
    term_part = term_part.rstrip("*")
    try:
        return float(term_part)
    except ValueError:
        for const_dict in const_list:
            for const_name, const_val in const_dict.items():
                if const_name in term_part:
                    term_part = term_part.replace(const_name, str(const_val))
        try:
            return float(eval(term_part))
        except Exception:
            print(f"Debug: Failed to parse coefficient from {term_part}")
            return 1.0


def generate_all_derivatives(max_deriv_orders):
    """Generate all derivative combinations implied by the max derivative orders."""
    all_derivatives = []
    for max_orders in max_deriv_orders:
        ranges = [range(order) for order in max_orders]
        if all(current_range.stop > 0 for current_range in ranges):
            all_derivatives.append([list(combo) for combo in product(*ranges)])
            continue
        all_derivatives.append([[0] * len(max_orders)])
    return all_derivatives


def get_deriv_name(var, deriv_tuple, spatial_vars):
    """Create the generated derivative symbol name."""
    if all(order == 0 for order in deriv_tuple):
        return var.upper()
    name = var.upper()
    for dim, count in enumerate(deriv_tuple):
        if count > 0:
            name += "_" + spatial_vars[dim] * count
    return name
