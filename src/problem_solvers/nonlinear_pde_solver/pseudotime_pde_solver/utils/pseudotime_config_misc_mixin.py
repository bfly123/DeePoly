class PseudotimeConfigMiscMixin:
    def _load_visualization_config(self):
        """Load visualization configuration with backward compatibility."""
        from src.abstract_class.config.visualization_config import VisualizationConfig

        vis_attr = getattr(self, "visualization", {})
        if isinstance(vis_attr, VisualizationConfig):
            return
        vis_dict = vis_attr if isinstance(vis_attr, dict) else {}
        legacy_fields = {
            "realtime_visualization": getattr(self, "realtime_visualization", None),
            "save_animation": getattr(self, "save_animation", None),
            "animation_skip": getattr(self, "animation_skip", None),
            "outputs": getattr(self, "outputs", None),
            "reference_policy": getattr(self, "reference_policy", None),
            "dataset_mode": getattr(self, "dataset_mode", None),
            "figure_dpi": getattr(self, "figure_dpi", None),
            "figure_format": getattr(self, "figure_format", None),
            "colormap": getattr(self, "colormap", None),
            "show_training_points": getattr(self, "show_training_points", None),
            "error_log_scale": getattr(self, "error_log_scale", None),
        }
        for key, value in legacy_fields.items():
            if value is not None and key not in vis_dict:
                vis_dict[key] = value
        self.visualization = VisualizationConfig.from_dict(vis_dict)
        self.realtime_visualization = self.visualization.realtime_visualization
        self.save_animation = self.visualization.save_animation
        self.animation_skip = self.visualization.animation_skip
        print(
            f"✓ Visualization config: mode={self.visualization.dataset_mode.value}, "
            f"outputs={self.visualization.outputs}"
        )

    def _int_list_fields(self):
        """Additional integer-list-like fields."""
        base_fields = super()._int_list_fields()
        return base_fields + [
            "max_iterations",
            "check_interval",
            "print_interval",
            "save_interval",
            "residual_stagnation_steps",
            "epochs_adam",
            "epochs_lbfgs",
            "max_retries",
            "spotter_skip",
        ]

    def validate(self) -> bool:
        """Validate pseudo-time configuration ranges."""
        if self.max_iterations <= 0:
            print(f"Error: max_iterations must be positive, got {self.max_iterations}")
            return False
        if self.spotter_skip <= 0:
            print(f"Error: spotter_skip must be positive, got {self.spotter_skip}")
            return False
        if self.residual_tol <= 0:
            print(f"Error: residual_tol must be positive, got {self.residual_tol}")
            return False
        if self.convergence_primary_tol <= 0:
            print(
                "Error: convergence_primary_tol must be positive, "
                f"got {self.convergence_primary_tol}"
            )
            return False
        if not (0 < self.convergence_improvement_factor <= 1.0):
            print(
                "Error: convergence_improvement_factor must be in (0, 1], "
                f"got {self.convergence_improvement_factor}"
            )
            return False
        if self.convergence_plateau_steps <= 0:
            print(
                "Error: convergence_plateau_steps must be positive, "
                f"got {self.convergence_plateau_steps}"
            )
            return False
        if self.pseudo_time_direction == 0:
            print("Error: pseudo_time_direction must be non-zero")
            return False
        if str(self.nonlinear_linearization).lower() not in {"picard", "newton"}:
            print(
                "Error: nonlinear_linearization must be 'picard' or 'newton', "
                f"got {self.nonlinear_linearization}"
            )
            return False
        if self.dt_min <= 0:
            print(f"Error: dt_min must be positive, got {self.dt_min}")
            return False
        if not (self.dt_min <= self.dt_init <= self.dt_max):
            print("Error: dt_init must satisfy dt_min <= dt_init <= dt_max")
            print(f"  Got: dt_min={self.dt_min}, dt_init={self.dt_init}, dt_max={self.dt_max}")
            return False
        return True

    def get_summary(self):
        """Get configuration summary."""
        summary = super().get_summary()
        summary.update(
            {
                "pseudo_time_stepping": {
                    "max_iterations": self.max_iterations,
                    "residual_tol": self.residual_tol,
                    "dt_init": self.dt_init,
                    "dt_range": [self.dt_min, self.dt_max],
                    "adaptive_dt": self.adaptive_dt,
                    "integrator": f"{self.time_scheme} (fixed backend)",
                    "pseudo_time_direction": self.pseudo_time_direction,
                    "nonlinear_linearization": self.nonlinear_linearization,
                    "convergence_primary_var": self.convergence_primary_var,
                    "convergence_primary_tol": self.convergence_primary_tol,
                    "convergence_plateau_steps": self.convergence_plateau_steps,
                    "convergence_plateau_tol": self.convergence_plateau_tol,
                    "rollback_on_plateau": self.rollback_on_plateau,
                    "reference_stable_single_retrain": self.reference_stable_single_retrain,
                    "reference_stable_retrain_stop_tol": self.reference_stable_retrain_stop_tol,
                    "reference_stable_adaptive_tail": self.reference_stable_adaptive_tail,
                    "pseudo_time_formulation": self.pseudo_time_formulation,
                },
                "initial_condition": self.initial_condition,
                "spotter": self.spotter,
            }
        )
        return summary
