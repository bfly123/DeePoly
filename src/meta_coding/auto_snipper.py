"""Facade for multi-operator equation parsing."""

from typing import Any, Dict, List, Optional

from src.meta_coding.auto_snipper_derivative_mixin import OperatorParserDerivativeMixin
from src.meta_coding.auto_snipper_term_mixin import OperatorParserTermMixin


class OperatorParser(OperatorParserDerivativeMixin, OperatorParserTermMixin):
    """Multi-operator equation parser."""

    def __init__(
        self,
        variables: List[str],
        spatial_vars: List[str],
        constants: Optional[Dict[str, float]] = None,
    ):
        self.variables = variables
        self.spatial_vars = spatial_vars
        self.constants = constants or {}


def parse_operators(
    operators: Dict[str, List[str]],
    variables: List[str],
    spatial_vars: List[str],
    constants: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    parser = OperatorParser(variables, spatial_vars, constants)
    return parser.parse(operators)


def parse_equation_to_list(
    equations: List[str],
    eq_nonlinear: List[str],
    variables: List[str],
    spatial_vars: List[str],
    constants: List[str],
) -> tuple:
    operators = {
        "linear": equations,
        "nonlinear": eq_nonlinear,
    }
    constants_dict = {const: 1.0 for const in constants} if constants else {}
    result = parse_operators(operators, variables, spatial_vars, constants_dict)
    deriv_orders = []
    max_deriv_orders = result["max_derivative_orders"]
    all_derivatives = result["derivatives"]
    return (
        [],
        deriv_orders,
        max_deriv_orders,
        [],
        all_derivatives,
    )
