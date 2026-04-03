from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


SUPPORTED_CONFIG_FILENAMES: tuple[str, ...] = ("config.yaml", "config.yml", "config.json")


def supported_config_filenames() -> Iterable[str]:
    return SUPPORTED_CONFIG_FILENAMES


def detect_config_format(path: str | os.PathLike[str]) -> str:
    return Path(path).expanduser().resolve().suffix.lower().lstrip(".")


def resolve_case_config_path(case_dir: Optional[str], *, filename: str | None = None) -> str:
    base_dir = Path(case_dir or os.getcwd()).expanduser().resolve()
    if filename is not None:
        candidate = base_dir / filename
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Configuration file not found: {candidate}")

    found = [base_dir / name for name in SUPPORTED_CONFIG_FILENAMES if (base_dir / name).exists()]
    if not found:
        expected = ", ".join(str(base_dir / name) for name in SUPPORTED_CONFIG_FILENAMES)
        raise FileNotFoundError(f"Configuration file not found. Tried: {expected}")
    if len(found) > 1:
        raise FileExistsError(
            "Multiple case config files found. Keep only one of: "
            + ", ".join(path.name for path in found)
        )
    return str(found[0])


def discover_case_dirs(case_root: str | os.PathLike[str]) -> list[Path]:
    root = Path(case_root).expanduser().resolve()
    candidates = {
        path.parent.resolve()
        for name in SUPPORTED_CONFIG_FILENAMES
        for path in root.rglob(name)
    }
    return sorted(candidates)


def load_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    suffix = config_path.suffix.lower()
    with open(config_path, "r", encoding="utf-8") as handle:
        if suffix == ".json":
            return json.load(handle)
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
            return dict(data or {})
    raise ValueError(f"Unsupported config format: {config_path}")


def save_config_file(
    path: str | os.PathLike[str],
    payload: dict[str, Any],
    *,
    sort_keys: bool = False,
) -> None:
    config_path = Path(path).expanduser().resolve()
    suffix = config_path.suffix.lower()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        if suffix == ".json":
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=sort_keys)
            handle.write("\n")
            return
        if suffix in {".yaml", ".yml"}:
            yaml.safe_dump(
                payload,
                handle,
                allow_unicode=True,
                sort_keys=sort_keys,
                default_flow_style=False,
            )
            return
    raise ValueError(f"Unsupported config format: {config_path}")
