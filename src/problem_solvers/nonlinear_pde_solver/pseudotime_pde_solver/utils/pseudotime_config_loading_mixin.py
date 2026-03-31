import json
import os

from src.meta_coding.auto_spotter import update_physics_loss_from_config


class PseudotimeConfigLoadingMixin:
    def _load_pseudo_time_config(self):
        """Load pseudo-time stepping parameters from nested or top-level config."""
        pts_params, config_source = self._get_pseudo_time_params()
        if hasattr(self, "T") or hasattr(self, "dt"):
            print("⚠ Warning: Found 'T' or 'dt' in general_pde config (will be ignored)")
            print("  For general_pde, use 'pseudo_time' block instead:")
            print('  "pseudo_time": {"max_iterations": ..., "dt_init": ..., ...}')

        self.max_iterations = pts_params.get("max_iterations", getattr(self, "max_iterations", self.max_iterations))
        self.residual_tol = pts_params.get("residual_tol", getattr(self, "residual_tol", self.residual_tol))
        self.dt_init = pts_params.get("dt_init", getattr(self, "dt_init", self.dt_init))
        self.dt_min = pts_params.get("dt_min", getattr(self, "dt_min", self.dt_min))
        self.dt_max = pts_params.get("dt_max", getattr(self, "dt_max", self.dt_max))
        self.adaptive_dt = pts_params.get("adaptive_dt", getattr(self, "adaptive_dt", self.adaptive_dt))
        self.dt_increase_factor = pts_params.get(
            "dt_increase_factor",
            getattr(self, "dt_increase_factor", self.dt_increase_factor),
        )
        self.dt_decrease_factor = pts_params.get(
            "dt_decrease_factor",
            getattr(self, "dt_decrease_factor", self.dt_decrease_factor),
        )
        self.check_interval = pts_params.get("check_interval", getattr(self, "check_interval", self.check_interval))
        self.print_interval = pts_params.get("print_interval", getattr(self, "print_interval", self.print_interval))
        self.save_interval = getattr(self, "save_interval", self.save_interval)
        self.max_residual = pts_params.get("max_residual", getattr(self, "max_residual", self.max_residual))
        self.residual_stagnation_steps = pts_params.get(
            "residual_stagnation_steps",
            getattr(self, "residual_stagnation_steps", self.residual_stagnation_steps),
        )
        self.convergence_primary_var = pts_params.get(
            "convergence_primary_var",
            getattr(self, "convergence_primary_var", self.convergence_primary_var),
        )
        self.convergence_primary_tol = pts_params.get(
            "convergence_primary_tol",
            getattr(self, "convergence_primary_tol", self.convergence_primary_tol),
        )
        self.convergence_improvement_factor = pts_params.get(
            "convergence_improvement_factor",
            getattr(self, "convergence_improvement_factor", self.convergence_improvement_factor),
        )
        self.convergence_plateau_steps = pts_params.get(
            "convergence_plateau_steps",
            getattr(self, "convergence_plateau_steps", self.convergence_plateau_steps),
        )
        self.convergence_plateau_tol = pts_params.get(
            "convergence_plateau_tol",
            getattr(self, "convergence_plateau_tol", self.convergence_plateau_tol),
        )
        self.convergence_dt_growth_tol = pts_params.get(
            "convergence_dt_growth_tol",
            getattr(self, "convergence_dt_growth_tol", self.convergence_dt_growth_tol),
        )
        self.rollback_on_plateau = pts_params.get(
            "rollback_on_plateau",
            getattr(self, "rollback_on_plateau", self.rollback_on_plateau),
        )
        self.rollback_dt_min = pts_params.get(
            "rollback_dt_min",
            getattr(self, "rollback_dt_min", self.rollback_dt_min),
        )
        self.force_retrain_on_plateau = pts_params.get(
            "force_retrain_on_plateau",
            getattr(self, "force_retrain_on_plateau", self.force_retrain_on_plateau),
        )
        self.return_best_on_exit = pts_params.get(
            "return_best_on_exit",
            getattr(self, "return_best_on_exit", self.return_best_on_exit),
        )
        self.reference_stable_single_retrain = pts_params.get(
            "reference_stable_single_retrain",
            getattr(
                self,
                "reference_stable_single_retrain",
                self.reference_stable_single_retrain,
            ),
        )
        self.reference_stable_retrain_stop_tol = pts_params.get(
            "reference_stable_retrain_stop_tol",
            getattr(
                self,
                "reference_stable_retrain_stop_tol",
                self.reference_stable_retrain_stop_tol,
            ),
        )
        self.reference_stable_adaptive_tail = pts_params.get(
            "reference_stable_adaptive_tail",
            getattr(
                self,
                "reference_stable_adaptive_tail",
                self.reference_stable_adaptive_tail,
            ),
        )
        if "time_scheme" in pts_params:
            requested_scheme = str(pts_params["time_scheme"])
            if requested_scheme != getattr(self, "time_scheme", "implicit_euler"):
                print(
                    "⚠ Warning: pseudo_time.time_scheme is ignored for pseudo-time solver; "
                    "using fixed implicit_euler backend"
                )
        if "save_interval" in pts_params:
            print(
                "⚠ Warning: pseudo_time.save_interval is currently not used by the "
                "pseudo-time iteration loop and will be ignored"
            )
        self.time_scheme = "implicit_euler"
        self.steady_state_mode = pts_params.get("steady_state_mode", getattr(self, "steady_state_mode", True))
        self.pseudo_time_formulation = pts_params.get(
            "pseudo_time_formulation",
            getattr(self, "pseudo_time_formulation", self.pseudo_time_formulation),
        )
        self.pseudo_time_direction = float(
            pts_params.get(
                "pseudo_time_direction",
                getattr(self, "pseudo_time_direction", self.pseudo_time_direction),
            )
        )
        self.nonlinear_linearization = str(
            pts_params.get(
                "nonlinear_linearization",
                getattr(self, "nonlinear_linearization", self.nonlinear_linearization),
            )
        ).lower()
        self.mass_eq_mask = pts_params.get(
            "mass_eq_mask",
            getattr(self, "mass_eq_mask", self.mass_eq_mask),
        )
        self.mass_var_map = pts_params.get(
            "mass_var_map",
            getattr(self, "mass_var_map", self.mass_var_map),
        )
        self.frozen_nonlinear_jacobian = pts_params.get(
            "frozen_nonlinear_jacobian",
            getattr(self, "frozen_nonlinear_jacobian", self.frozen_nonlinear_jacobian),
        )
        self._apply_reference_stable_defaults(pts_params)
        self._print_pseudo_time_summary(config_source)

    def _get_pseudo_time_params(self):
        pts_params = getattr(self, "pseudo_time", {})
        config_source = "nested 'pseudo_time'"
        if not pts_params:
            pts_params = getattr(self, "pseudo_time_stepping", {})
            config_source = "'pseudo_time_stepping' (legacy)" if pts_params else "default values"
        return pts_params, config_source

    def _print_pseudo_time_summary(self, config_source: str):
        print(f"✓ Pseudo-time stepping parameters (from {config_source}):")
        print(f"  integrator: {self.time_scheme} (fixed backend)")
        print(f"  max_iterations: {self.max_iterations}")
        print(f"  residual_tol: {self.residual_tol:.2e}")
        print(f"  dt_init: {self.dt_init}, dt_range: [{self.dt_min}, {self.dt_max}]")
        print(f"  adaptive_dt: {self.adaptive_dt}")
        print(
            "  convergence: "
            f"var[{self.convergence_primary_var}]<={self.convergence_primary_tol:.2e} "
            f"and rel<={self.residual_tol:.2e}"
        )
        print(
            "  plateau control: "
            f"steps={self.convergence_plateau_steps}, tol={self.convergence_plateau_tol:.2e}, "
            f"rollback={self.rollback_on_plateau}"
        )
        print(f"  reference_stable_single_retrain: {self.reference_stable_single_retrain}")
        print(f"  reference_stable_retrain_stop_tol: {self.reference_stable_retrain_stop_tol:.2e}")
        print(f"  reference_stable_adaptive_tail: {self.reference_stable_adaptive_tail}")
        mode = "(Snipper稳态求解)" if self.steady_state_mode else "(Snipper时间推进)"
        print(f"  solver_mode: {self.steady_state_mode} {mode}")
        print(f"  pseudo_time_formulation: {self.pseudo_time_formulation}")
        print(f"  pseudo_time_direction: {self.pseudo_time_direction:+.1f} (u_tau = dir * R)")
        print(f"  nonlinear_linearization: {self.nonlinear_linearization}")
        if self.mass_eq_mask:
            print(f"  mass_eq_mask: {self.mass_eq_mask}")
        if self.mass_var_map:
            print(f"  mass_var_map: {self.mass_var_map}")
        print(f"  frozen_nonlinear_jacobian: {self.frozen_nonlinear_jacobian}")
        if self.steady_state_mode and self.pseudo_time_formulation == "default":
            print(
                "  Warning: default + steady_state_mode=true degenerates to a frozen-nonlinear "
                "steady solve; dt/pseudo_time_direction do not enter the Snipper linear system. "
                "Use steady_state_mode=false for true pseudo-time mass stepping."
            )

    def _apply_reference_stable_defaults(self, pts_params):
        if self.pseudo_time_formulation != "reference_stable":
            return

        if "dt_increase_factor" not in pts_params:
            self.dt_increase_factor = 2.0
        if "force_retrain_on_plateau" not in pts_params:
            self.force_retrain_on_plateau = False
        if "rollback_dt_min" not in pts_params:
            self.rollback_dt_min = max(self.rollback_dt_min, 1e-2)

    def _load_initial_condition_config(self):
        """Load initial condition defaults."""
        default_ic = {"method": "given"}
        if not hasattr(self, "initial_condition") or not self.initial_condition:
            self.initial_condition = default_ic
        elif isinstance(self.initial_condition, dict):
            for key, value in default_ic.items():
                self.initial_condition.setdefault(key, value)
        print(f"Initial condition method: {self.initial_condition.get('method', 'given')}")

    def _normalize_spotter_block(self, raw_block):
        if not isinstance(raw_block, dict):
            return {}

        normalized = {}
        aliases = {
            "train_every": ("spotter_skip", "train_every"),
            "epochs_adam": ("epochs", "epochs_adam"),
            "epochs_lbfgs": ("epochs_lbfgs",),
            "max_retries": ("max_retries",),
            "tol": ("DNNtol", "tol"),
        }
        for target_key, source_keys in aliases.items():
            for source_key in source_keys:
                if source_key in raw_block:
                    normalized[target_key] = raw_block[source_key]
        return normalized

    def _load_spotter_config(self):
        """Load pseudo-time network-training config with legacy fallbacks."""
        spotter_cfg = {
            "train_every": getattr(self, "spotter_skip", 1),
            "epochs_adam": getattr(self, "epochs_adam", 5000),
            "epochs_lbfgs": getattr(self, "epochs_lbfgs", 0),
            "max_retries": getattr(self, "max_retries", 1),
            "tol": getattr(self, "DNNtol", 1e-4),
        }
        sources = ["main config"]

        fitter_training = getattr(self, "fitter_training", {})
        if isinstance(fitter_training, dict) and fitter_training:
            spotter_cfg.update(self._normalize_spotter_block(fitter_training))
            sources.append("fitter_training")
        else:
            self.fitter_training = {}

        raw_spotter = getattr(self, "spotter", {})
        if isinstance(raw_spotter, dict) and raw_spotter:
            spotter_cfg.update(self._normalize_spotter_block(raw_spotter))
            sources.append("spotter")
        else:
            self.spotter = {}

        self.spotter_skip = int(spotter_cfg["train_every"])
        self.epochs_adam = int(spotter_cfg["epochs_adam"])
        self.epochs_lbfgs = int(spotter_cfg["epochs_lbfgs"])
        self.max_retries = int(spotter_cfg["max_retries"])
        self.DNNtol = float(spotter_cfg["tol"])
        self.spotter = {
            "train_every": self.spotter_skip,
            "epochs_adam": self.epochs_adam,
            "epochs_lbfgs": self.epochs_lbfgs,
            "max_retries": self.max_retries,
            "tol": self.DNNtol,
        }

        source = " -> ".join(sources)
        print(f"Spotter training parameters (from {source}):")
        print(f"  train_every: {self.spotter_skip}")
        print(f"  epochs_adam: {self.epochs_adam}")
        print(f"  epochs_lbfgs: {self.epochs_lbfgs}")
        print(f"  max_retries: {self.max_retries}")
        print(f"  tol: {self.DNNtol:.2e}")

    def _load_fitter_training_config(self):
        """Backward-compatible alias for legacy callers/tests."""
        self._load_spotter_config()

    def _auto_code(self):
        """Run unified auto-code generation and disable the flag afterward."""
        if not getattr(self, "auto_code", False):
            return
        try:
            config_path = os.path.join(self.case_dir, "config.json")
            update_physics_loss_from_config(config_path)
            self._disable_auto_code(config_path)
            print("Auto code completed, please check the generated files and restart the program")
            import sys

            sys.exit(0)
        except Exception as exc:
            print(f"Auto-code generation failed: {exc}")

    @staticmethod
    def _disable_auto_code(config_path: str):
        """Set auto_code=false in the JSON config file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config["auto_code"] = False
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"Auto set auto_code to false in file: {config_path}")
        except Exception as exc:
            print(f"Cannot update configuration file: {exc}")
