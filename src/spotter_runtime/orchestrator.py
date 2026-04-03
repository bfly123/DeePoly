from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from src.spotter_runtime.config_support import resolve_python_executable, resolve_spotter_runtime
from src.spotter_runtime.registry import create_candidate_provider
from src.spotter_runtime.selector import (
    SpotterCandidateReport,
    SpotterSelectionReport,
    score_provider_candidate,
    select_best_candidate,
)
from src.spotter_runtime.types import SpotterCandidateConfig, SpotterRuntimeConfig


class SpotterOrchestrator:
    def __init__(
        self,
        *,
        config,
        model,
        data_train: Optional[Dict[str, Any]] = None,
        data_test: Optional[Dict[str, Any]] = None,
        reference_solution: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.model = model
        self.data_train = data_train
        self.data_test = data_test
        self.reference_solution = reference_solution
        self.runtime_config = self._build_runtime_config()
        self.providers: Dict[str, Any] = {}
        self.provider_init_errors: Dict[str, str] = {}
        self.selected_provider = None
        self.selection_report: Optional[SpotterSelectionReport] = None

    def initialize(self):
        candidates = self.runtime_config.enabled_candidates()
        if not candidates:
            raise ValueError("No enabled spotter candidates found in configuration")
        self.providers, self.provider_init_errors = self._build_candidate_providers(candidates)
        if not self.providers:
            error_summary = "; ".join(
                f"{name}: {error}" for name, error in self.provider_init_errors.items()
            )
            raise RuntimeError(
                "Failed to initialize any enabled spotter candidates"
                + (f": {error_summary}" if error_summary else "")
            )
        self.reselect()
        return self.selected_provider

    def reselect(self):
        candidates = self.runtime_config.enabled_candidates()
        reports = []
        for candidate in candidates:
            provider = self.providers.get(candidate.name)
            if provider is None:
                reports.append(
                    SpotterCandidateReport(
                        name=candidate.name,
                        backend=candidate.backend,
                        runtime=candidate.runtime,
                        score=float("inf"),
                        metric_used="error",
                        status="error",
                        error=self.provider_init_errors.get(candidate.name, "Provider initialization failed"),
                    )
                )
                continue
            try:
                report = score_provider_candidate(
                    provider=provider,
                    candidate=candidate,
                    selector_cfg=self.runtime_config.selector,
                    data_train=self.data_train,
                    data_test=self.data_test,
                    reference_solution=self.reference_solution,
                )
            except Exception as exc:
                report = SpotterCandidateReport(
                    name=candidate.name,
                    backend=candidate.backend,
                    runtime=candidate.runtime,
                    score=float("inf"),
                    metric_used="error",
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            reports.append(report)

        best_report = select_best_candidate(reports)
        if best_report.name not in self.providers:
            raise RuntimeError(
                best_report.error or "No usable spotter candidate available after initialization"
            )
        for report in reports:
            report.selected = report.name == best_report.name
        self.selected_provider = self.providers[best_report.name]
        self.selection_report = SpotterSelectionReport(
            mode=self.runtime_config.mode,
            selected_name=best_report.name,
            selected_backend=best_report.backend,
            metric=best_report.metric_used,
            candidates=reports,
        )
        return self.selected_provider

    def save_report(self, results_dir: str | Path) -> None:
        if self.selection_report is None:
            return
        path = Path(results_dir) / "spotter_selection_report.json"
        path.write_text(
            json.dumps(self.selection_report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def sync_model_state(self, *, model=None):
        if model is not None:
            self.model = model
        for provider in self.providers.values():
            refresh_fn = getattr(provider, "refresh_from_model", None)
            if callable(refresh_fn):
                refresh_fn(self.model)
        return self.reselect()

    def close(self) -> None:
        for provider in self.providers.values():
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                close_fn()
        self.providers = {}

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def _build_candidate_providers(self, candidates: list[SpotterCandidateConfig]) -> tuple[Dict[str, Any], Dict[str, str]]:
        provider_errors: Dict[str, str] = {}
        if self.runtime_config.mode == "parallel_select" and len(candidates) > 1:
            providers: Dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
                futures = {
                    executor.submit(
                        create_candidate_provider,
                        candidate=candidate,
                        config=self.config,
                        model=self.model,
                        data_train=self.data_train,
                        data_test=self.data_test,
                    ): candidate
                    for candidate in candidates
                }
                for future in as_completed(futures):
                    candidate = futures[future]
                    try:
                        providers[candidate.name] = future.result()
                    except Exception as exc:
                        provider_errors[candidate.name] = f"{type(exc).__name__}: {exc}"
            return providers, provider_errors

        providers: Dict[str, Any] = {}
        for candidate in candidates:
            try:
                providers[candidate.name] = create_candidate_provider(
                    candidate=candidate,
                    config=self.config,
                    model=self.model,
                    data_train=self.data_train,
                    data_test=self.data_test,
                )
            except Exception as exc:
                provider_errors[candidate.name] = f"{type(exc).__name__}: {exc}"
        return providers, provider_errors

    def _build_runtime_config(self) -> SpotterRuntimeConfig:
        raw = getattr(self.config, "spotter", {}) or {}
        candidate_dicts = []
        if isinstance(raw, dict):
            candidate_dicts = raw.get("candidates", []) or []
        if not raw or not candidate_dicts:
            backend = str(getattr(self.config, "spotter_feature_backend", "torch") or "torch")
            provider_cfg = dict(getattr(self.config, "spotter_feature", {}) or {})
            runtime_raw = provider_cfg.pop("runtime", "auto")
            runtime = resolve_spotter_runtime(runtime_raw, backend=backend, provider_cfg=provider_cfg)
            python_raw = provider_cfg.pop("python", None) or provider_cfg.pop("python_executable", None)
            python_executable = resolve_python_executable(python_raw, backend=backend)
            allow_torch_training = bool(provider_cfg.get("allow_torch_training", backend == "torch"))
            return SpotterRuntimeConfig(
                mode="single",
                selector={"metric": "priority"},
                candidates=[
                    SpotterCandidateConfig(
                        name=f"{backend}_default",
                        backend=backend,
                        runtime=runtime,
                        enabled=True,
                        priority=0,
                        device=getattr(self.config, "device", "cpu"),
                        python_executable=python_executable,
                        provider_cfg=provider_cfg,
                        allow_torch_training=allow_torch_training,
                    )
                ],
            )

        selector = dict(raw.get("selector", {}) or {})
        mode = str(raw.get("mode", "parallel_select") or "parallel_select")
        candidates = []
        for index, item in enumerate(candidate_dicts):
            if not isinstance(item, dict):
                continue
            provider_cfg = dict(item.get("feature", item.get("spotter_feature", {})) or {})
            candidates.append(
                SpotterCandidateConfig(
                    name=str(item.get("name") or f"candidate_{index}"),
                    backend=str(item.get("backend", "torch") or "torch"),
                    runtime=resolve_spotter_runtime(
                        item.get("runtime", "auto"),
                        backend=str(item.get("backend", "torch") or "torch"),
                        provider_cfg=provider_cfg,
                    ),
                    enabled=bool(item.get("enabled", True)),
                    priority=int(item.get("priority", index)),
                    device=item.get("device", getattr(self.config, "device", "cpu")),
                    python_executable=resolve_python_executable(
                        item.get("python") or item.get("python_executable"),
                        backend=str(item.get("backend", "torch") or "torch"),
                    ),
                    provider_cfg=provider_cfg,
                    allow_torch_training=bool(item.get("allow_torch_training", False)),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "name",
                            "backend",
                            "runtime",
                            "enabled",
                            "priority",
                            "device",
                            "python",
                            "python_executable",
                            "feature",
                            "spotter_feature",
                            "allow_torch_training",
                        }
                    },
                )
            )
        return SpotterRuntimeConfig(mode=mode, selector=selector, candidates=candidates)
