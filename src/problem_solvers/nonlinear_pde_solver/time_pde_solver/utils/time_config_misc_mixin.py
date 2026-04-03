from src.abstract_class.config.config_io import load_config_file, resolve_case_config_path, save_config_file
from src.meta_coding.auto_spotter import update_physics_loss_from_config


class TimeConfigMiscMixin:
    def _auto_code(self):
        if hasattr(self, "auto_code") and self.auto_code:
            try:
                config_path = resolve_case_config_path(self.case_dir)
                update_physics_loss_from_config(config_path)
                self._disable_auto_code(config_path)
                print("Auto code completed, please check the generated files and restart the program")
                import sys
                sys.exit(0)
            except Exception as exc:
                print(f"Auto-code generation failed: {exc}")

    def _disable_auto_code(self, config_path: str):
        try:
            config = load_config_file(config_path)
            config["auto_code"] = False
            save_config_file(config_path, config)
            print(f"Auto set auto_code to false in file: {config_path}")
        except Exception as exc:
            print(f"Cannot update configuration file: {exc}")

    def _int_list_fields(self):
        return super()._int_list_fields() + [
            "epochs_adam",
            "epochs_lbfgs",
            "max_retries",
            "spotter_skip",
            "max_time_steps",
            "time_output_interval",
        ]
