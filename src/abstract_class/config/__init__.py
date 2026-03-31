"""
Configuration module exports with lazy loading.
"""

from importlib import import_module

__all__ = [
    "BaseConfig",
    "BaseDataGenerator",
    "BaseVisualizer",
]

_MODULE_MAP = {
    "BaseConfig": "src.abstract_class.config.base_config",
    "BaseDataGenerator": "src.abstract_class.config.base_data",
    "BaseVisualizer": "src.abstract_class.config.base_visualize",
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
