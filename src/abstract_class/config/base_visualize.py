import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import os


class BaseVisualizer(ABC):
    """基础可视化类"""

    def __init__(self, config):
        self.config = config
        self.n_dim = config.n_dim
        self.n_eqs = config.n_eqs
        self.Ns = np.prod(config.n_segments)

    @abstractmethod
    def plot_solution(
        self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None
    ) -> None:
        """绘制解"""
        pass

    def _create_figure(self, figsize: tuple = (10, 8)) -> plt.Figure:
        """创建图形"""
        plt.style.use("seaborn")
        return plt.figure(figsize=figsize)

    def _save_figure(self, fig: plt.Figure, save_path: str) -> None:
        """保存图形"""
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

    def _close_figure(self, fig: plt.Figure) -> None:
        """关闭图形"""
        plt.close(fig)

    def _get_segment_boundaries(self, data: Dict, segment_idx: int) -> Dict:
        """获取段的边界"""
        return {
            "x_min": data["x_min"][segment_idx],
            "x_max": data["x_max"][segment_idx],
        }

    def _normalize_to_physical(
        self, x_norm: np.ndarray, segment_idx: int
    ) -> np.ndarray:
        """将归一化坐标转换回物理坐标"""
        x_min = self.config.x_min[segment_idx]
        x_max = self.config.x_max[segment_idx]
        return x_norm * (x_max - x_min) + x_min
