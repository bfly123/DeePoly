import numpy as np
import matplotlib.pyplot as plt
from typing import List


class SegmentOverlayMixin:

    def add_segment_boundaries_1d(self, ax, n_segments: int = None, x_domain: tuple = None):
        if n_segments is None and hasattr(self.config, 'n_segments'):
            n_segments = self.config.n_segments[0] if isinstance(self.config.n_segments, (list, tuple)) else self.config.n_segments
        if x_domain is None and hasattr(self.config, 'x_domain'):
            x_domain = self.config.x_domain[0]

        if n_segments and x_domain:
            x_min, x_max = x_domain
            boundaries = np.linspace(x_min, x_max, n_segments + 1)
            for boundary in boundaries[1:-1]:
                ax.axvline(boundary, color='k', linestyle=':', alpha=0.5, linewidth=1)

    def add_segment_boundaries_2d(self, axes_list: List):
        if not hasattr(self.config, 'x_min') or not hasattr(self.config, 'x_max'):
            return

        for n in range(getattr(self, 'Ns', 0)):
            x_min = self.config.x_min[n]
            x_max = self.config.x_max[n]

            for ax in axes_list:
                ax.plot([x_min[0], x_max[0]], [x_min[1], x_min[1]], 'k--', alpha=0.4, linewidth=1)
                ax.plot([x_min[0], x_max[0]], [x_max[1], x_max[1]], 'k--', alpha=0.4, linewidth=1)
                ax.plot([x_min[0], x_min[0]], [x_min[1], x_max[1]], 'k--', alpha=0.4, linewidth=1)
                ax.plot([x_max[0], x_max[0]], [x_min[1], x_max[1]], 'k--', alpha=0.4, linewidth=1)
