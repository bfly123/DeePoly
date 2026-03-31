from src.meta_coding.auto_snipper import OperatorParser, parse_operators


def test_operator_parser_tracks_zero_and_higher_order_derivatives():
    parser = OperatorParser(["u"], ["x", "y"])
    derivatives = parser.extract_unique_derivatives(["u + diff(u,x) + diff(u,y,2)"])
    assert derivatives == [[0, 0], [0, 2], [1, 0]]


def test_operator_parser_parse_groups_derivative_indices_by_operator():
    result = parse_operators(
        {
            "L1": ["u + diff(u,x)"],
            "L2": ["diff(u,x,2)"],
        },
        ["u"],
        ["x"],
    )
    assert result["all_derivatives"]["U"] == [0, 0]
    assert result["all_derivatives"]["U_x"] == [0, 1]
    assert result["all_derivatives"]["U_xx"] == [0, 2]
    assert result["operator_terms"]["L1"][0]["derivative_indices"] == [0, 1]
    assert result["operator_terms"]["L2"][0]["derivative_indices"] == [2]


def test_operator_parser_extract_all_derivatives_keeps_standalone_variables():
    parser = OperatorParser(["u", "v"], ["x"])
    derivatives = parser.extract_all_derivatives(["u + diff(v,x)"])
    assert derivatives["U"] == [0, 0]
    assert derivatives["V"] == [1, 0]
    assert derivatives["V_x"] == [1, 1]
