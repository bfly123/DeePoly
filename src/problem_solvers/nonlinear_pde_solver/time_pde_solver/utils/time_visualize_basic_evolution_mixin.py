from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class TimePDEBasicEvolutionMixin:
    def _plot_evolution_step_1d(
        self,
        T: float,
        u: np.ndarray,
        x: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        if not self._ensure_realtime_figure():
            return
        self.ax.clear()
        sort_idx = np.argsort(x[:, 0])
        x_sorted = x[sort_idx, 0]
        u_sorted = u[sort_idx, 0] if u.ndim > 1 else u[sort_idx]

        self.ax.plot(x_sorted, u_sorted, "b-", linewidth=3, label=f"u(x, t={T:.3f})")
        self.ax.set_xlabel("x", fontsize=12)
        self.ax.set_ylabel("u", fontsize=12)
        self.ax.set_title(f"Solution at T = {T:.4f}", fontsize=14, fontweight="bold")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        if hasattr(self.config, "x_domain"):
            self.ax.set_xlim([self.config.x_domain[0][0], self.config.x_domain[0][1]])

        u_min, u_max = u_sorted.min(), u_sorted.max()
        margin = (u_max - u_min) * 0.1 if u_max != u_min else 0.1
        self.ax.set_ylim([u_min - margin, u_max + margin])
        self._add_segment_boundaries_1d([self.ax])
        self._finalize_evolution_draw(save_path)

    def _plot_evolution_step_2d(
        self,
        T: float,
        u: np.ndarray,
        x: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        if not self._ensure_realtime_figure():
            return
        x2d = x if x.ndim == 2 else x.reshape(-1, 2)
        u_vals = u[:, 0] if u.ndim > 1 else u

        if not hasattr(self, "_scatter2d") or self._scatter2d is None:
            self.ax.clear()
            self._scatter2d = self.ax.scatter(
                x2d[:, 0],
                x2d[:, 1],
                c=u_vals,
                cmap="RdBu_r",
                s=20,
                alpha=0.9,
                edgecolors="none",
            )
            self.ax.set_xlabel("x", fontsize=12)
            self.ax.set_ylabel("y", fontsize=12)
            self.ax.set_aspect("equal")
            if hasattr(self.config, "x_domain") and len(self.config.x_domain) >= 2:
                self.ax.set_xlim([self.config.x_domain[0][0], self.config.x_domain[0][1]])
                self.ax.set_ylim([self.config.x_domain[1][0], self.config.x_domain[1][1]])
            self._colorbar_2d = getattr(self, "_colorbar_2d", None)
            if self._colorbar_2d is None:
                self._colorbar_2d = plt.colorbar(self._scatter2d, ax=self.ax, shrink=0.8)
                self._colorbar_2d.set_label("u", fontsize=12)
        else:
            self._scatter2d.set_offsets(np.column_stack((x2d[:, 0], x2d[:, 1])))
            self._scatter2d.set_array(u_vals)
            umin, umax = float(np.min(u_vals)), float(np.max(u_vals))
            if umax > umin:
                self._scatter2d.set_clim(umin, umax)
            if hasattr(self, "_colorbar_2d") and self._colorbar_2d is not None:
                self._colorbar_2d.update_normal(self._scatter2d)

        self.ax.set_title(f"Solution at T = {T:.4f}", fontsize=14, fontweight="bold")
        self._finalize_evolution_draw(save_path)

    def _finalize_evolution_draw(self, save_path: Optional[str]) -> None:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
