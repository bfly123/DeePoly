from importlib import import_module

__all__ = [
    "NullSpotterModel",
    "SpotterCandidateConfig",
    "SpotterCandidateReport",
    "SpotterOrchestrator",
    "SpotterRuntimeConfig",
    "SpotterSelectionReport",
    "BaseSpotterSession",
    "LiveTorchSpotterSession",
    "ProviderSpotterSession",
    "OrchestratedSpotterSession",
    "create_spotter_session",
]

_MODULE_MAP = {
    "NullSpotterModel": "src.spotter_runtime.null_model",
    "SpotterOrchestrator": "src.spotter_runtime.orchestrator",
    "SpotterCandidateReport": "src.spotter_runtime.selector",
    "SpotterSelectionReport": "src.spotter_runtime.selector",
    "SpotterCandidateConfig": "src.spotter_runtime.types",
    "SpotterRuntimeConfig": "src.spotter_runtime.types",
    "BaseSpotterSession": "src.spotter_runtime.session",
    "LiveTorchSpotterSession": "src.spotter_runtime.session",
    "ProviderSpotterSession": "src.spotter_runtime.session",
    "OrchestratedSpotterSession": "src.spotter_runtime.session",
    "create_spotter_session": "src.spotter_runtime.session",
}


def __getattr__(name):
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
