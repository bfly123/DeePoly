import json
import os

from src.meta_coding.auto_spotter import update_physics_loss_from_config


class FixIterationConfigMiscMixin:
    def _auto_code(self):
        if hasattr(self, "auto_code") and self.auto_code:
            try:
                config_path = os.path.join(self.case_dir, "config.json")
                update_physics_loss_from_config(config_path)
                self._disable_auto_code(config_path)
                print("Auto code completed, please check the generated files and restart the program")
                import sys
                sys.exit(0)
            except Exception as exc:
                print(f"Auto-code generation failed: {exc}")

    def _disable_auto_code(self, config_path: str):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            config["auto_code"] = False
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=4, ensure_ascii=False)
            print(f"Auto set auto_code to false in file: {config_path}")
        except Exception as exc:
            print(f"Cannot update configuration file: {exc}")

    def _int_list_fields(self):
        return super()._int_list_fields() + [
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
        checks = [
            ("max_iterations", self.max_iterations > 0, "must be positive"),
            ("residual_tol", self.residual_tol > 0, "must be positive"),
            ("dt_min", self.dt_min > 0, "must be positive"),
        ]
        for field_name, is_valid, message in checks:
            if not is_valid:
                print(f"Error: {field_name} {message}, got {getattr(self, field_name)}")
                return False
        if not (self.dt_min <= self.dt_init <= self.dt_max):
            print("Error: dt_init must satisfy dt_min <= dt_init <= dt_max")
            print(f"  Got: dt_min={self.dt_min}, dt_init={self.dt_init}, dt_max={self.dt_max}")
            return False
        return True

    def get_summary(self):
        summary = super().get_summary()
        summary.update(
            {
                "pseudo_time_stepping": {
                    "max_iterations": self.max_iterations,
                    "residual_tol": self.residual_tol,
                    "dt_init": self.dt_init,
                    "dt_range": [self.dt_min, self.dt_max],
                    "adaptive_dt": self.adaptive_dt,
                    "time_scheme": self.time_scheme,
                },
                "initial_condition": self.initial_condition,
                "fitter_training": self.fitter_training,
            }
        )
        return summary
