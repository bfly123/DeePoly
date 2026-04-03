import os
from datetime import datetime
from typing import Any, Optional

import numpy as np

from src.abstract_class.config.config_io import load_config_file, resolve_case_config_path


def resolve_config_path(case_dir: Optional[str], filename: str = "config.json") -> str:
    if filename != "config.json":
        return os.path.join(case_dir or os.getcwd(), filename)
    return resolve_case_config_path(case_dir)


def load_json_file(path: str) -> dict[str, Any]:
    return load_config_file(path)


def build_export_metadata(device: str) -> dict[str, str]:
    return {
        "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device_info": device,
    }


def serialize_config_value(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (tuple, list)):
        return [serialize_config_value(item) for item in obj]
    if isinstance(obj, dict):
        return {key: serialize_config_value(value) for key, value in obj.items()}
    if hasattr(obj, "__dict__"):
        return serialize_config_value(obj.__dict__)
    return str(obj)


def convert_int_list(values: list[Any]) -> list[Any]:
    try:
        return [int(value) if isinstance(value, str) else value for value in values]
    except ValueError:
        return values


def convert_scalar_string(value: str) -> Any:
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value
