import os
from pathlib import Path
from typing import Any, Optional

from src.abstract_class.config.config_json_helpers import (
    build_export_metadata,
    convert_int_list,
    load_json_file,
    resolve_config_path,
    serialize_config_value,
)
from src.abstract_class.config.config_io import detect_config_format, save_config_file


class BaseConfigJsonMixin:
    def _configured_device_for_export(self) -> Any:
        return self.__dict__.get("device", getattr(self, "device", None))

    def export_to_json(self, filename: str = "config.json") -> None:
        config_dict = self._public_config_dict()
        config_dict.update(build_export_metadata(self._configured_device_for_export()))
        save_config_file(filename, serialize_config_value(config_dict))
        print(f"Configuration exported to {filename}")

    def export_effective_config(
        self,
        results_dir: str,
        filename: str = "effective_config.json",
        *,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        payload = serialize_config_value(self._public_config_dict())
        metadata = {
            **build_export_metadata(self._configured_device_for_export()),
            "source_config_path": getattr(self, "_loaded_config_path", None),
            "source_config_format": getattr(self, "_loaded_config_format", None),
        }
        if hasattr(self, "method"):
            metadata["method"] = getattr(self, "method")
        if hasattr(self, "nn_backend"):
            metadata["nn_backend"] = getattr(self, "nn_backend")
        if hasattr(self, "problem_type"):
            metadata["problem_type"] = getattr(self, "problem_type")
        uses_snipper = getattr(self, "uses_snipper", None)
        if callable(uses_snipper):
            metadata["snipper_enabled"] = bool(uses_snipper())
        uses_pinns_only = getattr(self, "uses_pinns_only", None)
        if callable(uses_pinns_only) and uses_pinns_only():
            metadata["ignored_in_pinns_mode"] = ["n_segments", "poly_degree", "points_per_swap"]
        if extra_metadata:
            metadata.update(serialize_config_value(extra_metadata))
        payload["_effective_config"] = metadata

        target_path = Path(results_dir).expanduser().resolve() / filename
        save_config_file(target_path, payload, sort_keys=True)
        return str(target_path)

    def load_config_from_json(self, case_dir: Optional[str] = None) -> bool:
        config_path = resolve_config_path(case_dir)
        if not os.path.exists(config_path):
            return False
        try:
            self._loaded_config_path = os.path.abspath(config_path)
            self._loaded_config_format = detect_config_format(config_path)
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
