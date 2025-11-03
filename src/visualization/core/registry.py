from typing import Dict, Type, Optional
from .base import PlotComponent


class PlotRegistry:

    _registry: Dict[str, Type[PlotComponent]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(component_class: Type[PlotComponent]):
            if not issubclass(component_class, PlotComponent):
                raise TypeError(f"{component_class} must inherit from PlotComponent")
            if name in cls._registry:
                raise ValueError(f"Component '{name}' already registered")
            cls._registry[name] = component_class
            return component_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[PlotComponent]:
        if name not in cls._registry:
            raise KeyError(f"Component '{name}' not registered")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, component_name: Optional[str] = None, **config) -> PlotComponent:
        component_class = cls.get(name)
        instance_name = component_name or name
        return component_class(name=instance_name, config=config)

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._registry

    @classmethod
    def list_components(cls) -> list:
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
