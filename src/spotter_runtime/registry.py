from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from src.abstract_class.spotter_feature_provider import create_spotter_feature_provider_from_settings
from src.spotter_runtime.config_support import resolve_spotter_runtime
from src.spotter_runtime.runtime import SubprocessSpotterFeatureProvider
from src.spotter_runtime.types import SpotterCandidateConfig


def create_candidate_provider(
    *,
    candidate: SpotterCandidateConfig,
    config,
    model,
    data_train: Optional[Dict[str, Any]] = None,
    data_test: Optional[Dict[str, Any]] = None,
):
    runtime = resolve_spotter_runtime(
        candidate.runtime,
        backend=candidate.backend,
        provider_cfg=candidate.provider_cfg,
    )
    if runtime == "subprocess":
        repo_root = str(Path(__file__).resolve().parents[2])
        python_executable = candidate.python_executable or sys.executable
        startup_payload = {
            "repo_root": repo_root,
            "case_dir": getattr(config, "case_dir", None),
            "config": dict(getattr(config, "__dict__", {})),
            "candidate": asdict(candidate),
            "data_train": data_train,
            "data_test": data_test,
        }
        provider = SubprocessSpotterFeatureProvider(
            backend=candidate.backend,
            python_executable=python_executable,
            startup_payload=startup_payload,
            repo_root=repo_root,
        )
    else:
        provider = create_spotter_feature_provider_from_settings(
            backend=candidate.backend,
            provider_cfg=candidate.provider_cfg,
            case_dir=getattr(config, "case_dir", None),
            config=config,
            model=model,
            data_train=data_train,
            data_test=data_test,
            device=candidate.device or getattr(config, "device", "cpu"),
        )
    provider.candidate_name = candidate.name
    provider.allow_torch_training = bool(candidate.allow_torch_training)
    provider.runtime_mode = runtime
    return provider
