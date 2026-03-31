from pathlib import Path

import numpy as np

from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.solver import (
    PseudoTimePDESolver,
)
from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.utils import (
    PseudoTimePDEConfig,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "cases" / "nonlinear_pde_cases" / "pseudotime_pde_cases" / "lid_driven_cavity_re100"
CASE_DIR_RE400 = ROOT / "cases" / "nonlinear_pde_cases" / "pseudotime_pde_cases" / "lid_driven_cavity_re400"
CASE_DIR_RE1000 = ROOT / "cases" / "nonlinear_pde_cases" / "pseudotime_pde_cases" / "lid_driven_cavity_re1000"


def test_lid_driven_cavity_config_loads_as_three_equation_general_pde():
    config = PseudoTimePDEConfig(case_dir=str(CASE_DIR))

    assert config.problem_type == "general_pde"
    assert config.vars_list == ["u", "v", "p"]
    assert config.n_eqs == 3
    assert config.steady_state_mode is True
    assert config.pseudo_time_formulation == "reference_stable"
    assert config.mass_eq_mask == [1, 2]
    assert config.mass_var_map == [-1, 0, 1]
    assert config.max_iterations == 100
    assert config.dt_init == 0.1
    assert config.dt_min == 0.01
    assert config.dt_max == 0.1
    assert config.n_segments == [6, 6]
    assert not hasattr(config, "segment_mask_mode")
    assert not hasattr(config, "boundary_sampling_mode")
    assert config.swap_constraint_weight == 5.0
    assert config.spotter_skip == 10
    assert config.spotter["train_every"] == 10
    assert config.spotter["epochs_adam"] == 1200
    assert config.spotter["epochs_lbfgs"] == 0
    assert config.spotter["max_retries"] == 0
    assert config.max_retries == 0
    assert config.spotter["tol"] == 0.03
    assert config.convergence_primary_var == 0
    assert config.convergence_primary_tol == 1e-4
    assert config.convergence_plateau_steps == 10
    assert config.dt_increase_factor == 2.0
    assert config.rollback_on_plateau is True
    assert config.force_retrain_on_plateau is False
    assert len(config.eq_L1) == 3
    assert len(config.boundary_conditions) == 8
    assert isinstance(config.points_domain, int)
    assert config.points_domain == 6000
    assert config.points_domain_test == [400, 400]
    assert all(bc["points"] == 100 for bc in config.boundary_conditions)
    assert all(bc.get("var") in {"u", "v"} for bc in config.boundary_conditions)


def test_lid_driven_cavity_re1000_config_is_self_contained_unique_case():
    config = PseudoTimePDEConfig(case_dir=str(CASE_DIR_RE1000))

    assert config.problem_type == "general_pde"
    assert config.vars_list == ["u", "v", "p"]
    assert config.n_eqs == 3
    assert config.constants["Re"] == 1000
    assert np.isclose(config.constants["nu"], 0.001)
    assert config.initial_condition["method"] == "transfer"
    assert config.initial_condition["transfer"]["results_dir"] == "./results"
    assert config.initial_condition["transfer"]["solution_key"] == "sniper_train"
    assert config.initial_condition["transfer"]["load_coefficients"] is True
    assert config.initial_condition["transfer"]["skip_initial_training"] is True
    assert config.steady_state_mode is True
    assert config.pseudo_time_formulation == "reference_stable"
    assert config.max_iterations == 120
    assert config.reference_stable_single_retrain is False
    assert np.isclose(config.reference_stable_retrain_stop_tol, 1.0)
    assert config.n_segments == [6, 6]
    assert config.points_domain == 6000
    assert config.points_domain_test == [400, 400]
    assert config.adaptive_dt is False
    assert np.isclose(config.dt_init, 0.1)
    assert np.isclose(config.dt_min, 0.1)
    assert np.isclose(config.dt_max, 0.1)
    points_by_region = {(bc["region"], bc["var"]): bc["points"] for bc in config.boundary_conditions}
    assert points_by_region[("left", "u")] == 180
    assert points_by_region[("left", "v")] == 180
    assert points_by_region[("right", "u")] == 180
    assert points_by_region[("right", "v")] == 180
    assert points_by_region[("bottom", "u")] == 180
    assert points_by_region[("bottom", "v")] == 180
    assert points_by_region[("top", "u")] == 140
    assert points_by_region[("top", "v")] == 140
    assert config.spotter["epochs_adam"] == 1
    assert config.spotter["tol"] == 0.04


def test_lid_driven_cavity_re400_config_reuses_re100_transfer_setup():
    config = PseudoTimePDEConfig(case_dir=str(CASE_DIR_RE400))

    assert config.problem_type == "general_pde"
    assert config.vars_list == ["u", "v", "p"]
    assert config.n_eqs == 3
    assert config.constants["Re"] == 400
    assert np.isclose(config.constants["nu"], 0.0025)
    assert config.initial_condition["method"] == "transfer"
    assert config.initial_condition["transfer"]["source_case"] == "../lid_driven_cavity_re100"
    assert config.initial_condition["transfer"]["solution_key"] == "sniper_train"
    assert config.max_iterations == 220
    assert config.dt_init == 0.08
    assert config.dt_min == 0.01
    assert config.dt_max == 0.08
    assert np.isclose(config.reference_stable_retrain_stop_tol, 0.002)
    assert config.spotter["epochs_adam"] == 2400
    assert config.spotter["tol"] == 0.035
    assert all(bc["points"] == 120 for bc in config.boundary_conditions)

def test_lid_driven_cavity_solver_initialization_builds_multi_equation_components():
    solver = PseudoTimePDESolver(case_dir=str(CASE_DIR))

    assert solver.model.out_dim == 3
    assert solver.data_train["x"].shape[1] == 2
    assert solver.data_train["x"].shape[0] == solver.config.points_domain
    assert solver.data_test["x"].shape[1] == 2
    assert solver.data_test["x"].shape[0] == 400 * 400
    assert set(solver.data_train["global_boundary_dict"].keys()) == {0, 1, 2}
    assert solver.data_train["global_boundary_dict"][0]["dirichlet"]["x"].shape[0] == 400
    assert solver.data_train["global_boundary_dict"][1]["dirichlet"]["x"].shape[0] == 400
    assert solver.data_train["global_boundary_dict"][2]["dirichlet"]["x"].size == 0

    solver.fitter.fitter_init(solver.model)

    first_segment_features = solver.fitter._features[0]
    assert isinstance(first_segment_features, list)
    assert len(first_segment_features) > 0

    zero_solution = np.zeros((solver.data_train["x"].shape[0], solver.config.n_eqs))
    zero_segments = solver.fitter.global_to_segments(zero_solution)
    jacobian, rhs = solver.fitter._build_segment_jacobian(0, U_n_seg=zero_segments, dt=solver.config.dt_init)

    n_points = len(solver.data_train["x_segments_norm"][0])
    assert jacobian.shape == (n_points * solver.config.n_eqs, solver.config.n_eqs * solver.fitter.dgN)
    assert rhs.shape == (n_points * solver.config.n_eqs,)

    features0 = first_segment_features[0]
    L1 = solver.fitter._linear_operators[0]["L1"]
    mass_u = solver.fitter._build_mass_block(features0, 0, solver.config.n_eqs, solver.fitter.dgN)

    eq0 = jacobian[:n_points]
    eq1 = jacobian[n_points : 2 * n_points]

    np.testing.assert_allclose(eq0, solver.config.dt_init * L1[0], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(eq1, mass_u + solver.config.dt_init * L1[1], atol=1e-10, rtol=1e-10)


def test_lid_driven_cavity_direct_evaluation_dataset_includes_boundaries_without_duplicates():
    solver = PseudoTimePDESolver(case_dir=str(CASE_DIR))

    data = solver._build_direct_evaluation_dataset()
    coords = np.asarray(data["x"])
    unique_coords = np.unique(np.round(coords, decimals=12), axis=0)

    assert coords.shape == (400 * 400, 2)
    assert unique_coords.shape[0] == coords.shape[0]
    assert np.isclose(coords[:, 0].min(), 0.0)
    assert np.isclose(coords[:, 0].max(), 1.0)
    assert np.isclose(coords[:, 1].min(), 0.0)
    assert np.isclose(coords[:, 1].max(), 1.0)
