"""
Compact operator factory examples.

This module intentionally keeps only focused, runnable demonstrations. The
previous monolithic experiment script mixed examples, exploratory tests, and
long-form notes into a single 2k+ line file, which made it a structural
hotspot. The examples below preserve the most useful entry points while keeping
the module small and maintainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.abstract_class.operator_factory import create_operator_factory


@dataclass
class DemoBundle:
    all_derivatives: Dict[str, List[int]]
    constants: Dict[str, float]
    operator_terms: Dict[str, List[Dict]]
    features: List[np.ndarray]
    coeffs: np.ndarray


def _build_simple_demo_bundle() -> DemoBundle:
    all_derivatives = {
        "U": [0, 0],
        "U_X": [0, 1],
        "V": [1, 0],
        "V_X": [1, 1],
    }
    constants = {"c1": 2.0, "c2": 3.0}
    operator_terms = {
        "L1": [
            {"derivative_indices": [1], "symbolic_expr": "c1 * U_X", "var_idx": 0},
            {"derivative_indices": [1], "symbolic_expr": "c2 * V_X", "var_idx": 1},
        ],
        "N": [
            {"derivative_indices": [0], "symbolic_expr": "U", "var_idx": 0},
            {"derivative_indices": [0], "symbolic_expr": "V", "var_idx": 1},
        ],
        "F": [
            {"derivative_indices": [0], "symbolic_expr": "U + V", "var_idx": 0},
        ],
    }
    features = [
        np.array([[1.0, 0.0], [0.5, 1.5]]),
        np.array([[0.1, 0.2], [0.3, 0.4]]),
    ]
    coeffs = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    return DemoBundle(all_derivatives, constants, operator_terms, features, coeffs)


def _build_multi_var_demo_bundle() -> DemoBundle:
    all_derivatives = {
        "U": [0, 0],
        "U_X": [0, 1],
        "U_Y": [0, 2],
        "V": [1, 0],
        "V_X": [1, 1],
        "V_Y": [1, 2],
        "W": [2, 0],
        "W_X": [2, 1],
        "W_Y": [2, 2],
    }
    constants = {"alpha": 1.0, "beta": 0.5, "gamma": 0.1}
    operator_terms = {
        "L1": [
            {"derivative_indices": [1], "symbolic_expr": "U_X", "var_idx": 0},
            {"derivative_indices": [1], "symbolic_expr": "V_X", "var_idx": 1},
            {"derivative_indices": [1], "symbolic_expr": "W_X", "var_idx": 2},
        ],
        "N": [
            {"derivative_indices": [0], "symbolic_expr": "alpha * U", "var_idx": 0},
            {"derivative_indices": [0], "symbolic_expr": "beta * V", "var_idx": 1},
            {"derivative_indices": [0], "symbolic_expr": "gamma * W", "var_idx": 2},
        ],
    }
    features = [
        np.array([[1.0, 0.0], [0.5, 1.5]]),
        np.array([[0.2, 0.3], [0.4, 0.5]]),
        np.array([[0.6, 0.7], [0.8, 0.9]]),
    ]
    coeffs = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ]
    )
    return DemoBundle(all_derivatives, constants, operator_terms, features, coeffs)


def _create_demo_factory(bundle: DemoBundle, optimized: bool = True):
    return create_operator_factory(
        all_derivatives=bundle.all_derivatives,
        constants=bundle.constants,
        optimized=optimized,
    )


def _summarize_linear_result(result: np.ndarray) -> Dict[str, tuple]:
    return {
        "shape": result.shape,
        "column_sums": tuple(np.sum(result, axis=(0, 1)).round(6)),
    }


def _summarize_nonlinear_result(results: List[np.ndarray]) -> List[Dict[str, tuple]]:
    summaries = []
    for item in results:
        arr = np.asarray(item)
        summaries.append(
            {
                "shape": arr.shape,
                "mean": float(np.mean(arr)),
            }
        )
    return summaries


def example_multi_var_usage():
    """Return a compact multi-variable operator bundle and demo data."""
    bundle = _build_multi_var_demo_bundle()
    factory = _create_demo_factory(bundle, optimized=True)
    ops = factory.create_all_operators(bundle.operator_terms)
    return ops, bundle.features, bundle.coeffs


def run_simple_multi_var_demo():
    """Run the simplest operator-factory example used in regression tests."""
    bundle = _build_simple_demo_bundle()
    factory = _create_demo_factory(bundle, optimized=True)
    return factory.create_all_operators(bundle.operator_terms)


def comprehensive_multi_var_test():
    """Inspect the compact multi-variable demo outputs."""
    ops, features, coeffs = example_multi_var_usage()
    return {
        "linear": _summarize_linear_result(ops["L1_func"](features)),
        "nonlinear": _summarize_nonlinear_result(
            ops["N_func"](features=features, coeffs=coeffs, segment_idx=0)
        ),
    }


def run_n_equation_system_demo(n_eqs: int, var_names: List[str]) -> bool:
    """Smoke-check that operator creation still scales with equation count."""
    all_derivatives = {}
    operator_terms = {"L1": [], "N": []}
    for idx in range(n_eqs):
        name = var_names[idx].upper()
        all_derivatives[name] = [idx, 0]
        all_derivatives[f"{name}_X"] = [idx, 1]
        operator_terms["L1"].append(
            {"derivative_indices": [1], "symbolic_expr": f"{name}_X", "var_idx": idx}
        )
        operator_terms["N"].append(
            {"derivative_indices": [0], "symbolic_expr": f"{name}", "var_idx": idx}
        )

    features = [np.ones((3, 2)), np.full((3, 2), 0.5)]
    coeffs = np.ones((2, n_eqs, 2))
    ops = create_operator_factory(all_derivatives, optimized=True).create_all_operators(operator_terms)
    linear = ops["L1_func"](features)
    nonlinear = ops["N_func"](features=features, coeffs=coeffs, segment_idx=0)
    return linear.shape[0] == n_eqs and len(nonlinear) == n_eqs


def inspect_unified_operators(
    ops: Dict,
    features: List[np.ndarray],
    coeffs: np.ndarray,
    n_eqs: int,
    var_names: List[str],
):
    """Return compact metadata for operator inspection."""
    del n_eqs, var_names
    report = {}
    if "L1_func" in ops:
        report["L1_func"] = _summarize_linear_result(ops["L1_func"](features))
    if "N_func" in ops:
        report["N_func"] = _summarize_nonlinear_result(
            ops["N_func"](features=features, coeffs=coeffs, segment_idx=0)
        )
    return report


def unified_multi_equation_test():
    """Run smoke checks for 2-5 equation systems."""
    configs = [
        (2, ["u", "v"]),
        (3, ["u", "v", "p"]),
        (4, ["u", "v", "p", "t"]),
        (5, ["u", "v", "w", "p", "t"]),
    ]
    return {n_eqs: run_n_equation_system_demo(n_eqs, names) for n_eqs, names in configs}


def demonstrate_unified_approach():
    """Return a concise summary of the compact unified examples."""
    return {
        "supported_systems": unified_multi_equation_test(),
        "notes": [
            "Unified prediction keeps the operator API stable.",
            "Optimized and non-optimized factories share the same high-level contract.",
        ],
    }


def run_base_fitter_dimension_demo():
    """Return shape-level compatibility data for operator outputs."""
    bundle = _build_simple_demo_bundle()
    factory = _create_demo_factory(bundle, optimized=False)
    ops = factory.create_all_operators(bundle.operator_terms)
    linear = ops["L1_func"](bundle.features)
    nonlinear = ops["N_func"](features=bundle.features, coeffs=bundle.coeffs, segment_idx=0)
    return {
        "linear_shape": linear.shape,
        "nonlinear_shapes": [np.asarray(item).shape for item in nonlinear],
    }


def run_single_segment_multi_equation_demo():
    """Return a single-segment multi-equation operator snapshot."""
    ops, features, coeffs = example_multi_var_usage()
    return inspect_unified_operators(ops, features, coeffs, 3, ["u", "v", "w"])


def verify_base_fitter_compatibility():
    """Compact compatibility report for downstream fitters."""
    data = run_base_fitter_dimension_demo()
    return {
        "compatible": data["linear_shape"][0] >= 1 and len(data["nonlinear_shapes"]) >= 1,
        "details": data,
    }


def run_precompiled_system_demo():
    """Compare optimized and non-optimized operator outputs on the same bundle."""
    bundle = _build_simple_demo_bundle()
    basic_ops = _create_demo_factory(bundle, optimized=False).create_all_operators(bundle.operator_terms)
    optimized_ops = _create_demo_factory(bundle, optimized=True).create_all_operators(bundle.operator_terms)
    basic_linear = basic_ops["L1_func"](bundle.features)
    optimized_linear = optimized_ops["L1_func"](bundle.features)
    return {
        "linear_match": bool(np.allclose(basic_linear, optimized_linear)),
        "linear_shape": basic_linear.shape,
    }


def demonstrate_optimization_complete():
    """Return a short end-state summary for the compact examples."""
    return {
        "simple_demo": run_simple_multi_var_demo().keys(),
        "precompiled_demo": run_precompiled_system_demo(),
        "compatibility": verify_base_fitter_compatibility(),
    }


def quick_usage_example():
    """Return the compact usage narrative for docs and manual runs."""
    return "\n".join(
        [
            "1. Create a factory with create_operator_factory(...).",
            "2. Build operators with factory.create_all_operators(...).",
            "3. Call linear operators with features and nonlinear operators with explicit keywords.",
        ]
    )


# Backward-compatible aliases retained for the example compatibility shim.
test_simple_multi_var = run_simple_multi_var_demo
test_n_equation_system = run_n_equation_system_demo
test_operators_unified = inspect_unified_operators
test_correct_dimensions_base_fitter = run_base_fitter_dimension_demo
test_single_segment_multi_equation = run_single_segment_multi_equation_demo
test_precompiled_system = run_precompiled_system_demo


if __name__ == "__main__":
    print("Simple demo keys:", sorted(run_simple_multi_var_demo().keys()))
    print("Unified systems:", unified_multi_equation_test())
    print("Precompiled check:", run_precompiled_system_demo())
