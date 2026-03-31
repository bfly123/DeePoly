import os
from typing import Any, Optional

from src.abstract_class.config.config_json_helpers import (
    convert_int_list,
    convert_scalar_string,
    load_json_file,
    resolve_config_path,
)


class BasePDEConfigLoadingMixin:
    def load_config_from_json_relaxed(self, case_dir: Optional[str] = None) -> bool:
        config_path = resolve_config_path(case_dir)
        if not os.path.exists(config_path):
            return False
        try:
            config_dict = load_json_file(config_path)
            print(f"Successfully loaded configuration from {config_path}")
            self._assign_relaxed_config_values(config_dict)
            return True
        except Exception as exc:
            print(f"Error loading configuration: {exc}")
            return False

    def _assign_relaxed_config_values(self, config_dict: dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if key in {"export_time", "device_info"}:
                continue
            setattr(self, key, self._coerce_relaxed_value(key, value))

    def _coerce_relaxed_value(self, key: str, value: Any) -> Any:
        if isinstance(value, list):
            return self._coerce_relaxed_list_value(key, value)
        if isinstance(value, str):
            return convert_scalar_string(value)
        return value

    def _coerce_relaxed_list_value(self, key: str, value: list[Any]) -> Any:
        if key in self._int_list_fields():
            return convert_int_list(value)
        if key in self._list_fields():
            return self._process_list_field(key, value)
        return value
