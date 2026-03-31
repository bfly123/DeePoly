from pathlib import Path

from tools.compare_lid_driven_cavity_regression import build_regression_report


ROOT = Path(__file__).resolve().parents[1]
CURRENT = (
    ROOT
    / "cases"
    / "nonlinear_pde_cases"
    / "pseudotime_pde_cases"
    / "lid_driven_cavity_re100"
    / "results"
    / "solution_data.npz"
)
REFERENCE = Path("/home/bfly/workspace/computeforcfd/混合网络/2D_NS_Stable/u_nRe100.npy")


def test_lid_driven_cavity_regression_report_builds_from_current_and_reference_results():
    report = build_regression_report(CURRENT, REFERENCE)

    assert report["reference_shape"][1] == 3
    assert "spotter_test" in report["comparisons"]
    assert "sniper_test" in report["comparisons"]

    for source_name in ["spotter_test", "sniper_test"]:
        for var_name in ["u", "v", "p"]:
            metrics = report["comparisons"][source_name][var_name]
            assert set(metrics["quantile_abs_diff"]) == {"q05", "q50", "q95"}
            for value in metrics["quantile_abs_diff"].values():
                assert value >= 0.0
