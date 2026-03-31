import json
import os
from typing import Any, Optional

from src.abstract_class.config.config_json_helpers import (
    build_export_metadata,
    convert_int_list,
    load_json_file,
    resolve_config_path,
    serialize_config_value,
)


class BaseConfigJsonMixin:
    def export_to_json(self, filename: str = "config.json") -> None:
        config_dict = self._public_config_dict()
        config_dict.update(build_export_metadata(self.device))
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(serialize_config_value(config_dict), handle, indent=2)
        print(f"Configuration exported to {filename}")

    def load_config_from_json(self, case_dir: Optional[str] = None) -> bool:
        config_path = resolve_config_path(case_dir)
        if not os.path.exists(config_path):
            return False
        try:
            self._assign_known_config_values(load_json_file(config_path))
            return True
        except Exception as exc:
            print(f"Error loading configuration: {exc}")
            return False

    def _public_config_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }

    def _assign_known_config_values(self, config_dict: dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, self._coerce_known_config_value(key, value))

    def _coerce_known_config_value(self, key: str, value: Any) -> Any:
        if isinstance(value, list):
            if key in self._int_list_fields():
                return convert_int_list(value)
            if key in self._list_fields():
                return self._process_list_field(key, value)
        return value
