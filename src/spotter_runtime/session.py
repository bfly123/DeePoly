from __future__ import annotations

from typing import Any, Dict, Optional

from src.abstract_class.spotter_feature_provider import create_spotter_feature_provider
from src.spotter_runtime.config_support import resolve_spotter_runtime
from src.spotter_runtime.orchestrator import SpotterOrchestrator


class BaseSpotterSession:
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
        self.provider = None
        self.orchestrator = None

    def initialize(self):
        return self

    def feature_source(self):
        return self.model

    def active_backend_name(self) -> str:
        return "torch"

    def uses_external_backend(self) -> bool:
        return self.active_backend_name() != "torch"

    def allow_local_torch_training(self) -> bool:
        return not self.uses_external_backend()

    def can_train_externally(self) -> bool:
        return False

    def train_external_step(
        self,
        *,
        problem_type: str,
        dt: float,
        U_current,
        iteration: Optional[int] = None,
    ):
        raise RuntimeError("Active spotter session does not support external training.")

    def sync(self, *, model_updated: bool = False) -> None:
        _ = model_updated

    def save_report(self, results_dir: Optional[str]) -> None:
        _ = results_dir

    def close(self) -> None:
        close_targets = []
        if self.provider is not None:
            close_targets.append(self.provider)
        if self.orchestrator is not None:
            close_targets.append(self.orchestrator)
        seen = set()
        for target in close_targets:
            if id(target) in seen:
                continue
            seen.add(id(target))
            close_fn = getattr(target, "close", None)
            if callable(close_fn):
                close_fn()


class LiveTorchSpotterSession(BaseSpotterSession):
    def feature_source(self):
        return self.model

    def active_backend_name(self) -> str:
        return "torch"

    def allow_local_torch_training(self) -> bool:
        return True


class ProviderSpotterSession(BaseSpotterSession):
    def initialize(self):
        self.provider = create_spotter_feature_provider(
            self.config,
            self.model,
            data_train=self.data_train,
            data_test=self.data_test,
        )
        return self

    def feature_source(self):
        return self.provider or self.model

    def active_backend_name(self) -> str:
        source = self.provider
        if source is None:
            return str(getattr(self.config, "spotter_feature_backend", "torch") or "torch").lower()
        return str(getattr(source, "backend_name", "torch") or "torch").lower()

    def allow_local_torch_training(self) -> bool:
        source = self.provider
        if source is not None and hasattr(source, "allow_torch_training"):
            return bool(source.allow_torch_training)
        cfg = getattr(self.config, "spotter_feature", {}) or {}
        return bool(cfg.get("allow_torch_training", False))

    def can_train_externally(self) -> bool:
        source = self.provider
        if source is None:
            return False
        supports = getattr(source, "supports_external_training", None)
        if callable(supports):
            return bool(supports())
        return bool(supports)

    def train_external_step(
        self,
        *,
        problem_type: str,
        dt: float,
        U_current,
        iteration: Optional[int] = None,
    ):
        if self.provider is None:
            raise RuntimeError("External spotter provider is not initialized.")
        return self.provider.train_external_step(
            problem_type=problem_type,
            dt=dt,
            U_current=U_current,
            iteration=iteration,
        )

    def sync(self, *, model_updated: bool = False) -> None:
        if not model_updated or self.provider is None:
            return
        refresh_fn = getattr(self.provider, "refresh_from_model", None)
        if callable(refresh_fn):
            refresh_fn(self.model)


class OrchestratedSpotterSession(BaseSpotterSession):
    def initialize(self):
        self.orchestrator = SpotterOrchestrator(
            config=self.config,
            model=self.model,
            data_train=self.data_train,
            data_test=self.data_test,
            reference_solution=self.reference_solution,
        )
        self.provider = self.orchestrator.initialize()
        return self

    def feature_source(self):
        return self.provider or self.model

    def active_backend_name(self) -> str:
        source = self.provider
        if source is None:
            return str(getattr(self.config, "spotter_feature_backend", "torch") or "torch").lower()
        return str(getattr(source, "backend_name", "torch") or "torch").lower()

    def allow_local_torch_training(self) -> bool:
        source = self.provider
        if source is not None and hasattr(source, "allow_torch_training"):
            return bool(source.allow_torch_training)
        return False

    def can_train_externally(self) -> bool:
        source = self.provider
        if source is None:
            return False
        supports = getattr(source, "supports_external_training", None)
        if callable(supports):
            return bool(supports())
        return bool(supports)

    def train_external_step(
        self,
        *,
        problem_type: str,
        dt: float,
        U_current,
        iteration: Optional[int] = None,
    ):
        if self.provider is None:
            raise RuntimeError("No selected external spotter provider is available.")
        result = self.provider.train_external_step(
            problem_type=problem_type,
            dt=dt,
            U_current=U_current,
            iteration=iteration,
        )
        self.provider = self.orchestrator.reselect() if self.orchestrator is not None else self.provider
        return result

    def sync(self, *, model_updated: bool = False) -> None:
        if self.orchestrator is None:
            return
        if model_updated:
            self.provider = self.orchestrator.sync_model_state(model=self.model)
            return
        self.provider = self.orchestrator.reselect()

    def save_report(self, results_dir: Optional[str]) -> None:
        if self.orchestrator is None or not results_dir:
            return
        self.orchestrator.save_report(results_dir)


def create_spotter_session(
    *,
    config,
    model,
    data_train: Optional[Dict[str, Any]] = None,
    data_test: Optional[Dict[str, Any]] = None,
    reference_solution: Optional[Dict[str, Any]] = None,
):
    raw_spotter = getattr(config, "spotter", {}) or {}
    legacy_cfg = getattr(config, "spotter_feature", {}) or {}
    backend = str(getattr(config, "spotter_feature_backend", "torch") or "torch").lower()
    legacy_runtime = resolve_spotter_runtime(
        legacy_cfg.get("runtime", "auto"),
        backend=backend,
        provider_cfg=legacy_cfg,
    )

    if raw_spotter or legacy_runtime != "inprocess":
        return OrchestratedSpotterSession(
            config=config,
            model=model,
            data_train=data_train,
            data_test=data_test,
            reference_solution=reference_solution,
        ).initialize()

    if backend == "torch" and legacy_runtime == "inprocess":
        return LiveTorchSpotterSession(
            config=config,
            model=model,
            data_train=data_train,
            data_test=data_test,
            reference_solution=reference_solution,
        ).initialize()

    return ProviderSpotterSession(
        config=config,
        model=model,
        data_train=data_train,
        data_test=data_test,
        reference_solution=reference_solution,
    ).initialize()
