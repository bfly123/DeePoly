import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FigureContext:
    fig: plt.Figure
    ax: plt.Axes
    index: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PlotComponent(ABC):

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def render(self, context: FigureContext, data: Dict[str, Any]) -> None:
        pass

    def validate_data(self, data: Dict[str, Any]) -> bool:
        required_keys = self.get_required_keys()
        return all(key in data for key in required_keys)

    def get_required_keys(self) -> list:
        return []

    def set_config(self, **kwargs):
        self.config.update(kwargs)

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
