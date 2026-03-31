import json
import os
from datetime import datetime
from typing import Any, Optional

import numpy as np


def resolve_config_path(case_dir: Optional[str], filename: str = "config.json") -> str:
    return os.path.join(case_dir or os.getcwd(), filename)


def load_json_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
