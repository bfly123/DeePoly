#!/usr/bin/env python3
"""Run all case directories under cases/ and collect a regression summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-root",
        default="cases",
        help="Root directory containing case config.json files.",
    )
    parser.add_argument(
        "--output-root",
        default="tmp/case_regression",
        help="Directory for logs and summary outputs.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run `python -m src.cli`.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda", None],
        help="Optional CLI --device override.",
    )
    parser.add_argument(
        "--linear-device",
        default=None,
        choices=["auto", "cpu", "cuda", None],
        help="Optional CLI --linear-device override.",
    )
    parser.add_argument(
        "--skip-autocode",
        action="store_true",
        help="Pass --skip-autocode to each case run.",
    )
    parser.add_argument(
        "--case-filter",
        default="",
        help="Only run cases whose relative path contains this substring.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this case index after sorting, for resume/debug.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of cases to run after start-index.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=0,
        help="Optional per-case timeout. 0 means no timeout.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def discover_cases(case_root: Path, case_filter: str) -> list[Path]:
    cases = sorted(path.parent for path in case_root.rglob("config.json"))
    if case_filter:
        cases = [case for case in cases if case_filter in str(case.relative_to(repo_root()))]
    return cases


def build_run_command(
    case_dir: Path,
    python_exe: str,
    skip_autocode: bool,
    device: str | None,
    linear_device: str | None,
) -> list[str]:
    cmd = [python_exe, "-m", "src.cli", str(case_dir)]
    if skip_autocode:
        cmd.append("--skip-autocode")
    if device:
        cmd.extend(["--device", device])
    if linear_device:
        cmd.extend(["--linear-device", linear_device])
    return cmd


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def summarize_case(case_dir: Path, rel_case: str, result: subprocess.CompletedProcess[str], elapsed: float) -> dict[str, Any]:
    cfg = load_json(case_dir / "config.json")
    summary: dict[str, Any] = {
        "case": rel_case,
        "problem_type": cfg.get("problem_type"),
        "return_code": result.returncode,
        "elapsed_seconds": elapsed,
        "status": "passed" if result.returncode == 0 else "failed",
    }

    results_dir = case_dir / "results"
    accuracy_report = load_optional_json(results_dir / "accuracy_report.json")
    convergence_info = load_optional_json(results_dir / "convergence_info.json")
    regression_report = load_optional_json(results_dir / "regression_report.json")

    if accuracy_report:
        summary["spotter_rel_l2"] = ((accuracy_report.get("spotter") or {}).get("rel_l2"))
        summary["snipper_rel_l2"] = ((accuracy_report.get("snipper_direct") or {}).get("rel_l2"))
        fixed_point = accuracy_report.get("fixed_point") or {}
        summary["fixed_point_iterations"] = fixed_point.get("iterations")
        summary["fixed_point_final_residual"] = fixed_point.get("final_residual")

    if convergence_info:
        summary["converged"] = convergence_info.get("converged")
        summary["iterations"] = convergence_info.get("iterations")
        summary["final_residual"] = convergence_info.get("final_residual")
        summary["reason"] = convergence_info.get("reason")

    if regression_report:
        summary["regression_report"] = regression_report

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        tail_source = stderr if stderr else stdout
        tail_lines = tail_source.splitlines()[-20:]
        summary["error_tail"] = tail_lines

    return summary


def main() -> int:
    args = parse_args()
    root = repo_root()
    case_root = (root / args.case_root).resolve()
    output_root = (root / args.output_root).resolve()
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(case_root, args.case_filter)
    if args.start_index:
        cases = cases[args.start_index :]
    if args.limit > 0:
        cases = cases[: args.limit]

    manifest = [str(case.relative_to(root)) for case in cases]
    dump_json(output_root / "manifest.json", {"cases": manifest})
    print(f"Discovered {len(cases)} case(s)")

    rows: list[dict[str, Any]] = []
    for index, case_dir in enumerate(cases, start=1):
        rel_case = str(case_dir.relative_to(root))
        safe_name = rel_case.replace("/", "__")
        log_path = logs_dir / f"{safe_name}.log"
        print(f"[{index}/{len(cases)}] Running {rel_case}")
        cmd = build_run_command(
            case_dir=case_dir,
            python_exe=args.python,
            skip_autocode=args.skip_autocode,
            device=args.device,
            linear_device=args.linear_device,
        )
        start = time.time()
        try:
            completed = subprocess.run(
                cmd,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=(args.timeout_seconds or None),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.time() - start
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            log_path.write_text(
                (stdout if isinstance(stdout, str) else stdout.decode("utf-8", errors="replace"))
                + "\nSTDERR\n"
                + (stderr if isinstance(stderr, str) else stderr.decode("utf-8", errors="replace")),
                encoding="utf-8",
            )
            row = {
                "case": rel_case,
                "problem_type": load_json(case_dir / "config.json").get("problem_type"),
                "return_code": -999,
                "elapsed_seconds": elapsed,
                "status": "timeout",
                "error_tail": ["TimeoutExpired"],
            }
            rows.append(row)
            dump_json(output_root / "summary.json", rows)
            continue

        elapsed = time.time() - start
        log_path.write_text(
            (completed.stdout or "") + "\nSTDERR\n" + (completed.stderr or ""),
            encoding="utf-8",
        )
        row = summarize_case(case_dir, rel_case, completed, elapsed)
        rows.append(row)
        dump_json(output_root / "summary.json", rows)

    passed = sum(1 for row in rows if row["status"] == "passed")
    failed = sum(1 for row in rows if row["status"] == "failed")
    timeout = sum(1 for row in rows if row["status"] == "timeout")
    dump_json(
        output_root / "summary_stats.json",
        {
            "total": len(rows),
            "passed": passed,
            "failed": failed,
            "timeout": timeout,
        },
    )
    print(
        f"Completed regression sweep: total={len(rows)} passed={passed} failed={failed} timeout={timeout}"
    )
    return 0 if failed == 0 and timeout == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
