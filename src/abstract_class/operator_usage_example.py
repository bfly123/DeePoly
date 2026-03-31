"""
Compatibility wrapper for the operator usage examples.

The example implementation now lives under ``examples/`` so production
packages are not mixed with large demo and test-oriented scripts.
"""

from importlib import import_module


_EXAMPLE_MODULE = "examples.operator_usage_example"


def __getattr__(name):
    module = import_module(_EXAMPLE_MODULE)
    return getattr(module, name)


def __dir__():
    module = import_module(_EXAMPLE_MODULE)
    return sorted(set(module.__dict__.keys()))
