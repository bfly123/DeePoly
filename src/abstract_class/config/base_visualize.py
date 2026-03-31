import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from abc import ABC, abstractmethod
import os

from src.abstract_class.config.visualization_config import VisualizationConfig


def _import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})


class BaseVisualizer(ABC):

    def __init__(self, config=None):
        self.config = config
        if config:
            if hasattr(config, "visualization") and isinstance(config.visualization, dict):
                config.visualization = VisualizationConfig.from_dict(config.visualization)
            # 优先使用spatial_vars判断维度，其次使用n_dim
            spatial_vars = getattr(config, 'spatial_vars', None)
            if spatial_vars is not None:
                self.n_dim = len(spatial_vars)
            else:
                self.n_dim = getattr(config, 'n_dim', len(config.x_domain) if hasattr(config, 'x_domain') else 1)
            self.n_eqs = getattr(config, 'n_eqs', None)
            self.Ns = np.prod(getattr(config, 'n_segments', [1]))

    def _create_figure(self, figsize: tuple = (10, 8)) -> plt.Figure:
        return plt.figure(figsize=figsize)

    def _save_figure(self, fig: plt.Figure, save_path: str) -> None:
        if save_path:
            if os.path.dirname(save_path):  # 只有当路径包含目录时才创建
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')

    def _close_figure(self, fig: plt.Figure) -> None:
        plt.close(fig)

    def _to_numpy_cpu(self, data):
        if data is None:
            return None
        torch = _import_torch()
        if torch is not None and isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @abstractmethod
    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None):
        pass

    def render(self, results: Dict, save_dir: str, **kwargs):
        pass

    def plot_convergence(self, errors: list, save_path: Optional[str] = None):
        pass

    def update_animation_data(self, iteration: int, data: Dict):
        pass

    def finalize_animation(self, save_path: str):
        pass
