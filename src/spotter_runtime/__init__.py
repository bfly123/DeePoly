from importlib import import_module

__all__ = [
    "NullSpotterModel",
    "SpotterCandidateConfig",
    "SpotterCandidateReport",
    "SpotterOrchestrator",
    "SpotterRuntimeConfig",
    "SpotterSelectionReport",
]

_MODULE_MAP = {
    "NullSpotterModel": "src.spotter_runtime.null_model",
    "SpotterOrchestrator": "src.spotter_runtime.orchestrator",
    "SpotterCandidateReport": "src.spotter_runtime.selector",
    "SpotterSelectionReport": "src.spotter_runtime.selector",
    "SpotterCandidateConfig": "src.spotter_runtime.types",
    "SpotterRuntimeConfig": "src.spotter_runtime.types",
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
