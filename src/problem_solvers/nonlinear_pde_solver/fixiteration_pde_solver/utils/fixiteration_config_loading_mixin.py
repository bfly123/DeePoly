class FixIterationConfigLoadingMixin:
    def _load_pseudo_time_config(self):
        pts_params, config_source = self._select_pseudo_time_config()
        self._warn_if_time_fields_present()
        self._apply_pseudo_time_params(pts_params)
        print(f"✓ Pseudo-time stepping parameters (from {config_source}):")
        print(f"  time_scheme: {self.time_scheme}")
        print(f"  max_iterations: {self.max_iterations}")
        print(f"  residual_tol: {self.residual_tol:.2e}")
        print(f"  under_relaxation: {self.under_relaxation}")
        print(f"  dt_init: {self.dt_init}, dt_range: [{self.dt_min}, {self.dt_max}]")
        print(f"  adaptive_dt: {self.adaptive_dt}")

    def _select_pseudo_time_config(self):
        for attr_name, source in [
            ("fixiteration", "nested 'fixiteration'"),
            ("pseudo_time", "nested 'pseudo_time'"),
            ("pseudo_time_stepping", "'pseudo_time_stepping' (legacy)"),
        ]:
            params = getattr(self, attr_name, {})
            if params:
                return params, source
        return {}, "default values"

    def _warn_if_time_fields_present(self):
        if hasattr(self, "T") or hasattr(self, "dt"):
            print("⚠ Warning: Found 'T' or 'dt' in general_pde config (will be ignored)")
            print("  For general_pde, use 'pseudo_time' block instead:")
            print('  "pseudo_time": {"max_iterations": ..., "dt_init": ..., ...}')

    def _apply_pseudo_time_params(self, pts_params):
        field_names = [
            "max_iterations",
            "residual_tol",
            "under_relaxation",
            "dt_init",
            "dt_min",
            "dt_max",
            "adaptive_dt",
            "dt_increase_factor",
            "dt_decrease_factor",
            "check_interval",
            "print_interval",
            "save_interval",
            "max_residual",
            "residual_stagnation_steps",
        ]
        for field_name in field_names:
            setattr(self, field_name, pts_params.get(field_name, getattr(self, field_name)))
        if "time_scheme" in pts_params:
            self.time_scheme = pts_params["time_scheme"]

    def _load_initial_condition_config(self):
        default_ic = {
            "method": "spotter",
            "use_periodic": False,
            "spotter_config": {
                "epochs": 10000,
                "learning_rate": 0.01,
                "n_pde_points": 5000,
                "optimizer": "adam",
                "target_residual": 0.01,
                "print_interval": 200,
                "early_stopping": True,
                "patience": 500,
                "max_retries": 1,
            },
        }
        if not hasattr(self, "initial_condition") or not self.initial_condition:
            self.initial_condition = default_ic
        elif isinstance(self.initial_condition, dict):
            for key, value in default_ic.items():
                if key not in self.initial_condition:
                    self.initial_condition[key] = value
        if "spotter_config" not in self.initial_condition:
            self.initial_condition["spotter_config"] = default_ic["spotter_config"]
        print(f"Initial condition method: {self.initial_condition.get('method', 'spotter')}")
        spotter_cfg = self.initial_condition.get("spotter_config", {})
        print(f"  Spotter epochs: {spotter_cfg.get('epochs', 'N/A')}")
        print(f"  Spotter learning rate: {spotter_cfg.get('learning_rate', 'N/A')}")

    def _load_fitter_training_config(self):
        has_fitter_block = hasattr(self, "fitter_training") and self.fitter_training
        if not has_fitter_block:
            self.fitter_training = {}
        elif isinstance(self.fitter_training, dict):
            for source_key, target_key in [
                ("epochs", "epochs_adam"),
                ("epochs_lbfgs", "epochs_lbfgs"),
                ("DNNtol", "DNNtol"),
            ]:
                if source_key in self.fitter_training:
                    setattr(self, target_key, self.fitter_training[source_key])
        source = "fitter_training block" if has_fitter_block else "main config"
        print(f"Fitter training parameters (from {source}):")
        print(f"  epochs_adam: {self.epochs_adam}")
        print(f"  epochs_lbfgs: {self.epochs_lbfgs}")
        print(f"  DNNtol: {self.DNNtol:.2e}")
