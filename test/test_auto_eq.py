from src.meta_coding.auto_eq import parse_equation_to_list
from src.meta_coding.auto_eq_linear_parser import split_term_sign_and_division
from src.meta_coding.auto_eq_nonlinear_parser import parse_nonlinear_factors


def test_parse_equation_to_list_handles_linear_and_nonlinear_terms():
    result, deriv_orders, max_deriv_orders, nonlinear, all_derivatives = parse_equation_to_list(
        equation_str=[
            "diff(u,x) + diff(v,y)",
            "diff(p,x) - diff(u,x,2)/Re - diff(u,y,2)/Re",
        ],
        equation_nonlinear_str=[
            "0",
            "u*diff(u,x) + v*diff(u,y)",
        ],
        vars_list=["u", "v", "p"],
        spatial_vars=["x", "y"],
        const_list=[{"Re": 100}],
    )

    assert deriv_orders[0] == (0, 0)
    assert max_deriv_orders == [[2, 2], [0, 1], [1, 0]]
    assert result[0][0][-1] == "U_x"
    assert result[0][1][-1] == "V_y"
    assert result[1][0][-1] == "P_x"
    assert {item[-1] for item in nonlinear} == {"U", "U_x", "U_y", "V"}
    assert all_derivatives[0][0] == [0, 0]


def test_parse_equation_to_list_emits_const_term_when_present():
    result, _, _, nonlinear, _ = parse_equation_to_list(
        equation_str=["2*u + 3"],
        equation_nonlinear_str=[],
        vars_list=["u"],
        spatial_vars=["x"],
        const_list=[],
    )

    assert result == [[[2.0, 0, 0, "U"], [3.0, -1, -1, "const"]]]
    assert nonlinear == []


def test_linear_parser_handles_division_by_named_constant():
    coef, num_part = split_term_sign_and_division("-diff(u,x)/Re", [{"Re": 50}])
    assert coef == -0.02
    assert num_part == "diff(u,x)"


def test_nonlinear_parser_skips_numeric_and_constant_factors():
    context = {
        "n_dim": 1,
        "deriv_to_index": {(0,): 0, (1,): 1},
    }
    factors = parse_nonlinear_factors(
        "2*Re*u*diff(u,x)",
        ["u"],
        ["x"],
        [{"Re": 100}],
        context,
    )
    assert factors == [(0, 0), (0, 1)]
