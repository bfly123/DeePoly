import numpy as np
from types import SimpleNamespace

from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.core.fitter import (
    PseudoTimePDEFitter,
)


def build_helper_only_fitter():
    fitter = PseudoTimePDEFitter.__new__(PseudoTimePDEFitter)
    fitter.config = SimpleNamespace(
        n_eqs=1,
        steady_state_mode=True,
        pseudo_time_formulation="reference_stable",
        reference_stable_adaptive_tail=False,
    )
    fitter.adaptive_dt = True
    fitter.dt_init = 0.05
    fitter.dt_min = 0.01
    fitter.dt_max = 0.2
    fitter.dt_increase_factor = 2.0
    fitter.dt_decrease_factor = 0.5
    fitter.max_residual = 10.0
    fitter.residual_stagnation_steps = 3
    fitter.check_interval = 5
    fitter.residual_tol = 1e-4
    fitter.convergence_primary_var = 0
    fitter.convergence_primary_tol = 1e-4
    fitter.convergence_improvement_factor = 0.999
    fitter.convergence_plateau_steps = 3
    fitter.convergence_plateau_tol = 1e-2
    fitter.convergence_dt_growth_tol = 1e-3
    fitter.rollback_on_plateau = True
    fitter.rollback_dt_min = 0.01
    fitter.force_retrain_on_plateau = True
    fitter.return_best_on_exit = True
    fitter.residual_history = [1.0, 0.5]
    fitter.dt_history = [0.1, 0.05]
    fitter.solution_norm_history = [2.0, 1.5]
    fitter.primary_update_history = [0.2, 0.1]
    fitter.primary_relative_update_history = [0.2, 0.1]
    fitter.max_abs_update_history = [0.2, 0.1]
    fitter.best_primary_update = 0.1
    fitter.best_iteration = 1
    fitter.restart_count = 1
    fitter.restart_history = [{"iteration": 4, "dt_before": 0.05, "dt_after": 0.025}]
    return fitter


def test_pseudotime_residual_norm_uses_larger_reference_norm():
    fitter = build_helper_only_fitter()
    residual, solution_norm = fitter._compute_residual_norm(
        np.array([[2.0], [2.0]]),
        np.array([[1.0], [1.0]]),
    )
    assert np.isclose(solution_norm, np.linalg.norm(np.array([[2.0], [2.0]])))
    assert residual > 0.0


def test_pseudotime_convergence_metrics_include_primary_and_global_updates():
    fitter = build_helper_only_fitter()
    metrics = fitter._compute_convergence_metrics(
        np.array([[1.0], [2.0]]),
        np.array([[0.5], [1.5]]),
    )

    assert metrics["primary_var"] == 0
    assert metrics["primary_max_abs_update"] == 0.5
    assert metrics["max_abs_update"] == 0.5
    assert metrics["relative_l2_update"] > 0.0


def test_pseudotime_adaptive_dt_handles_high_mid_and_low_ranges():
    fitter = build_helper_only_fitter()
    dt_high, stagnation_high = fitter._update_adaptive_timestep(0.2, 0.1, 0)
    dt_mid, stagnation_mid = fitter._update_adaptive_timestep(0.05, 0.04, 1)
    dt_low, stagnation_low = fitter._update_adaptive_timestep(5e-4, 6e-4, 2)

    assert np.isclose(dt_high, fitter.dt_max)
    assert fitter.dt_min < dt_mid <= fitter.dt_max
    assert fitter.rollback_dt_min <= dt_low < dt_mid
    assert stagnation_high == 1
    assert stagnation_mid == 2
    assert stagnation_low == 0


def test_pseudotime_adaptive_dt_uses_dt_min_floor_for_nonsteady_modes():
    fitter = build_helper_only_fitter()
    fitter.config = SimpleNamespace(
        n_eqs=1,
        steady_state_mode=False,
        pseudo_time_formulation="reference_stable",
    )

    dt_low, _ = fitter._update_adaptive_timestep(5e-4, 6e-4, 0)

    assert np.isclose(dt_low, fitter.dt_min + (max(fitter.dt_max * fitter.ADAPTIVE_DT_SCALE_FACTOR, fitter.dt_min) - fitter.dt_min) * ((5e-4 - fitter.ADAPTIVE_DT_THRESHOLD_LOW) / (fitter.ADAPTIVE_DT_THRESHOLD_MID - fitter.ADAPTIVE_DT_THRESHOLD_LOW)))


def test_pseudotime_failure_checks_cover_divergence_and_stagnation():
    fitter = build_helper_only_fitter()
    failure = fitter._check_failure(20.0, 0, 2)
    assert failure == ("\nDiverged at iter 3: Res=2.00e+01 > 1.00e+01", "divergence")

    failure = fitter._check_failure(1.0, 4, 6)
    assert failure == ("\nStagnated at iter 7: No progress for 4 steps", "stagnation")

    assert fitter._check_failure(1.0, 1, 1) is None


def test_pseudotime_convergence_info_preserves_histories_for_terminal_states():
    fitter = build_helper_only_fitter()
    metrics = {
        "primary_max_abs_update": 5e-5,
        "primary_relative_update": 2e-5,
        "max_abs_update": 8e-5,
    }
    info = fitter._build_convergence_info(True, "", 8, 1e-6, 0.25, 1.2, metrics=metrics)
    assert info["converged"] is True
    assert info["final_tau"] == 0.25
    assert info["residual_history"] == fitter.residual_history
    assert info["primary_update_history"] == fitter.primary_update_history
    assert info["final_primary_update"] == 5e-5
    assert info["restart_count"] == 1

    stalled = fitter._build_convergence_info(False, "max_iterations", 10, 1e-2, 0.5, 2.4, metrics=metrics)
    assert stalled["reason"] == "max_iterations"
    assert stalled["dt_history"] == fitter.dt_history
    assert stalled["restart_history"] == fitter.restart_history


def test_pseudotime_plateau_restart_logic_matches_old_project_style():
    fitter = build_helper_only_fitter()
    assert fitter._is_metric_improved(0.09, 0.1) is True
    assert fitter._is_metric_improved(0.09995, 0.1) is False
    assert fitter._should_restart_on_plateau(0.02, 3) is True
    assert fitter._should_restart_on_plateau(5e-3, 3) is False
    assert fitter._get_plateau_restart_dt(0.04) == 0.02


def test_reference_stable_controller_keeps_dt_until_primary_update_is_small():
    fitter = build_helper_only_fitter()

    dt_same, stagnation_same = fitter._select_next_timestep(
        0.04,
        residual=0.08,
        residual_prev=0.07,
        stagnation_count=1,
        improved=False,
        primary_update=2e-2,
    )
    dt_grow, stagnation_grow = fitter._select_next_timestep(
        0.04,
        residual=8e-4,
        residual_prev=1e-3,
        stagnation_count=2,
        improved=True,
        primary_update=5e-4,
    )

    assert np.isclose(dt_same, 0.04)
    assert stagnation_same == 2
    assert np.isclose(dt_grow, 0.08)
    assert stagnation_grow == 0


def test_reference_stable_controller_can_switch_to_adaptive_tail():
    fitter = build_helper_only_fitter()
    fitter.config.reference_stable_adaptive_tail = True
    fitter.dt_min = 5e-4
    fitter.rollback_dt_min = 5e-4

    dt_tail, stagnation_tail = fitter._select_next_timestep(
        0.04,
        residual=5e-5,
        residual_prev=6e-5,
        stagnation_count=2,
        improved=True,
        primary_update=5e-4,
    )

    assert np.isclose(dt_tail, 0.02)
    assert stagnation_tail == 0


def test_reference_stable_adaptive_tail_waits_for_true_tail_residual():
    fitter = build_helper_only_fitter()
    fitter.config.reference_stable_adaptive_tail = True
    fitter.dt_min = 5e-4
    fitter.rollback_dt_min = 5e-4

    dt_same, stagnation_same = fitter._select_next_timestep(
        0.04,
        residual=8e-4,
        residual_prev=1e-3,
        stagnation_count=2,
        improved=True,
        primary_update=5e-4,
    )

    assert np.isclose(dt_same, 0.08)
    assert stagnation_same == 0
