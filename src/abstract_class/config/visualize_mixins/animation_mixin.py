import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Callable
from PIL import Image
import io
import os


class AnimationMixin:

    def create_animation(self, snapshots: List[np.ndarray], x: np.ndarray,
                        save_path: str, render_func: Optional[Callable] = None,
                        duration: int = 100, title_func: Optional[Callable] = None):
        frames = []

        for i, snapshot in enumerate(snapshots):
            if render_func:
                fig = render_func(x, snapshot, i)
            else:
                fig = self._default_render_frame(x, snapshot, i, title_func)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close(fig)

        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                      duration=duration, loop=0)

    def _default_render_frame(self, x: np.ndarray, u: np.ndarray, frame_idx: int,
                             title_func: Optional[Callable] = None):
        fig = self._create_figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        if x.shape[1] == 1:
            x_plot = x.squeeze()
            sort_idx = np.argsort(x_plot)
            ax.plot(x_plot[sort_idx], u.squeeze()[sort_idx], 'b-', linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('u')
        elif x.shape[1] == 2:
            from scipy.interpolate import griddata
            Xi, Yi, Ui = self.reconstruct_2d_grid(x, u)
            c = ax.contourf(Xi, Yi, Ui, 50, cmap='viridis')
            fig.colorbar(c, ax=ax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        if title_func:
            ax.set_title(title_func(frame_idx))
        else:
            ax.set_title(f'Frame {frame_idx}')

        ax.grid(True, alpha=0.3)
        return fig

    def create_convergence_animation(self, residual_history: List[float],
                                    solution_history: List[np.ndarray],
                                    x: np.ndarray, save_path: str, duration: int = 100):
        frames = []

        for i, (res, sol) in enumerate(zip(residual_history, solution_history)):
            fig = self._create_figure(figsize=(14, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.semilogy(range(i+1), residual_history[:i+1], 'r-', linewidth=2)
            ax1.scatter([i], [res], c='red', s=100, zorder=5)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Residual (log scale)')
            ax1.set_title(f'Convergence (Iter {i})')
            ax1.grid(True, alpha=0.3)

            if x.shape[1] == 1:
                x_plot = x.squeeze()
                sort_idx = np.argsort(x_plot)
                ax2.plot(x_plot[sort_idx], sol.squeeze()[sort_idx], 'b-', linewidth=2)
                ax2.set_xlabel('x')
                ax2.set_ylabel('u')
            elif x.shape[1] == 2:
                from scipy.interpolate import griddata
                Xi, Yi, Ui = self.reconstruct_2d_grid(x, sol)
                c = ax2.contourf(Xi, Yi, Ui, 50, cmap='viridis')
                fig.colorbar(c, ax=ax2)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')

            ax2.set_title('Current Solution')
            ax2.grid(True, alpha=0.3)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close(fig)

        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                      duration=duration, loop=0)
