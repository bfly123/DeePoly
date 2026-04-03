#!/usr/bin/env python3
"""Migrate DeePoly case configs from JSON to YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.abstract_class.config.config_io import (
    discover_case_dirs,
    load_config_file,
    resolve_case_config_path,
    save_config_file,
)


def _normalize_modern_config(payload: dict) -> dict:
    normalized = dict(payload)

    raw_method = str(normalized.get("method", "") or "").strip().lower()
    if raw_method == "hybrid":
        normalized["method"] = "deepoly"

    backend = (
        normalized.get("nn_backend")
        or normalized.get("spotter_backend")
        or normalized.get("spotter_feature_backend")
        or "torch"
    )
    normalized["nn_backend"] = str(backend).strip().lower()

    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help="Single case directory or a root directory containing multiple cases.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Treat path as a root directory and migrate all discovered case directories under it.",
    )
    parser.add_argument(
        "--case-filter",
        default="",
        help="Only migrate cases whose path contains this substring.",
    )
    parser.add_argument(
        "--output-name",
        default="config.yaml",
        choices=["config.yaml", "config.yml"],
        help="Target YAML filename.",
    )
    parser.add_argument(
        "--replace-json",
        action="store_true",
        help="Remove source config.json after successfully writing YAML.",
    )
    parser.add_argument(
        "--sort-keys",
        action="store_true",
        help="Sort keys when writing YAML. Default preserves existing insertion order.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize legacy keys such as method=hybrid or backend aliases.",
    )
    return parser.parse_args()


def iter_case_dirs(root: Path, recursive: bool, case_filter: str) -> list[Path]:
    if recursive:
        cases = discover_case_dirs(root)
    else:
        cases = [root]
    if case_filter:
        cases = [case for case in cases if case_filter in str(case)]
    return sorted(Path(case).resolve() for case in cases)


def migrate_case(
    case_dir: Path,
    output_name: str,
    replace_json: bool,
    sort_keys: bool,
    normalize: bool,
) -> tuple[str, str]:
    config_path = Path(resolve_case_config_path(case_dir))
    suffix = config_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        return ("skipped", f"{case_dir}: already using {config_path.name}")

    if config_path.name != "config.json":
        return ("skipped", f"{case_dir}: unsupported source config {config_path.name}")

    payload = load_config_file(config_path)
    if normalize:
        payload = _normalize_modern_config(payload)
    yaml_path = case_dir / output_name
    if yaml_path.exists():
        return ("skipped", f"{case_dir}: target already exists ({yaml_path.name})")

    save_config_file(yaml_path, payload, sort_keys=sort_keys)

    if replace_json:
        config_path.unlink()
        return ("migrated", f"{case_dir}: config.json -> {yaml_path.name} (json removed)")
    return ("migrated", f"{case_dir}: config.json -> {yaml_path.name}")


def main() -> int:
    args = parse_args()
    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"path not found: {root}")

    case_dirs = iter_case_dirs(root, args.recursive, args.case_filter)
    if not case_dirs:
        print("No case directories discovered.")
        return 1

    migrated = 0
    skipped = 0
    failed = 0

    for case_dir in case_dirs:
        try:
            status, message = migrate_case(
                case_dir=case_dir,
                output_name=args.output_name,
                replace_json=args.replace_json,
                sort_keys=args.sort_keys,
                normalize=not args.no_normalize,
            )
            print(message)
            if status == "migrated":
                migrated += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"{case_dir}: failed ({exc})")

    print(
        f"Migration summary: total={len(case_dirs)} migrated={migrated} skipped={skipped} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
