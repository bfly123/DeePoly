from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SpotterBackendSpec:
    name: str
    required_modules: Tuple[str, ...] = ()


BACKEND_REGISTRY: Dict[str, SpotterBackendSpec] = {
    'torch': SpotterBackendSpec(name='torch', required_modules=('torch',)),
    'jax': SpotterBackendSpec(name='jax', required_modules=('jax', 'optax')),
    'paddle': SpotterBackendSpec(name='paddle', required_modules=('paddle',)),
}


def get_backend_spec(backend: str) -> SpotterBackendSpec:
    backend_name = str(backend or 'torch').lower()
    return BACKEND_REGISTRY.get(backend_name, SpotterBackendSpec(name=backend_name))


def modules_available(module_names: Tuple[str, ...]) -> bool:
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception:
            return False
    return True


def backend_supports_inprocess(backend: str, *, required_modules: Tuple[str, ...] | None = None) -> bool:
    spec = get_backend_spec(backend)
    module_names = tuple(required_modules or spec.required_modules)
    return modules_available(module_names)
