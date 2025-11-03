import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from IPython import display


class RealtimeVisualizationMixin:

    def init_realtime_plot(self, layout: str = 'single') -> Dict:
        plt.ion()

        if layout == 'single':
            fig = self._create_figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            plot_state = {'fig': fig, 'ax': ax, 'lines': []}
        elif layout == 'dual':
            fig = self._create_figure(figsize=(14, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            plot_state = {'fig': fig, 'ax1': ax1, 'ax2': ax2, 'lines1': [], 'lines2': []}
        elif layout == 'quad':
            fig = self._create_figure(figsize=(14, 12))
            axes = [fig.add_subplot(221), fig.add_subplot(222),
                   fig.add_subplot(223), fig.add_subplot(224)]
            plot_state = {'fig': fig, 'axes': axes, 'lines': [[] for _ in axes]}

        return plot_state

    def update_realtime_plot(self, plot_state: Dict, data: Dict, clear: bool = True):
        if clear:
            for key in plot_state:
                if key.startswith('ax'):
                    ax = plot_state[key]
                    ax.clear()
                elif key == 'axes':
                    for ax in plot_state['axes']:
                        ax.clear()

        if 'ax' in plot_state:
            self._update_single_axis(plot_state['ax'], data)
        elif 'ax1' in plot_state:
            self._update_dual_axis(plot_state['ax1'], plot_state['ax2'], data)
        elif 'axes' in plot_state:
            self._update_multi_axis(plot_state['axes'], data)

        plot_state['fig'].canvas.draw()
        plot_state['fig'].canvas.flush_events()

    def _update_single_axis(self, ax, data: Dict):
        if 'x' in data and 'y' in data:
            ax.plot(data['x'], data['y'], data.get('style', 'b-'), linewidth=2)
        if 'title' in data:
            ax.set_title(data['title'])
        if 'xlabel' in data:
            ax.set_xlabel(data['xlabel'])
        if 'ylabel' in data:
            ax.set_ylabel(data['ylabel'])
        if data.get('semilogy', False):
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    def _update_dual_axis(self, ax1, ax2, data: Dict):
        if 'left' in data:
            self._update_single_axis(ax1, data['left'])
        if 'right' in data:
            self._update_single_axis(ax2, data['right'])

    def _update_multi_axis(self, axes: List, data: Dict):
        for i, ax in enumerate(axes):
            if f'plot_{i}' in data:
                self._update_single_axis(ax, data[f'plot_{i}'])

    def close_realtime_plot(self, plot_state: Dict):
        plt.ioff()
        plt.close(plot_state['fig'])

    def update_convergence_realtime(self, plot_state: Dict, iteration: int,
                                   residual: float, residuals: List[float],
                                   tau: Optional[float] = None, taus: Optional[List[float]] = None):
        ax1 = plot_state['ax1']
        ax2 = plot_state['ax2']

        ax1.clear()
        ax2.clear()

        ax1.semilogy(range(len(residuals)), residuals, 'b-', linewidth=2)
        ax1.scatter([len(residuals)-1], [residual], c='red', s=100, zorder=5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Residual (log scale)')
        ax1.set_title(f'Convergence (Iter {iteration}, Res={residual:.2e})')
        ax1.grid(True, alpha=0.3)

        if tau is not None and taus is not None:
            ax2.semilogy(taus, residuals, 'r-', linewidth=2)
            ax2.scatter([tau], [residual], c='blue', s=100, zorder=5)
            ax2.set_xlabel('Pseudo-time τ')
            ax2.set_ylabel('Residual (log scale)')
            ax2.set_title(f'τ = {tau:.2e}')
            ax2.grid(True, alpha=0.3)

        plot_state['fig'].canvas.draw()
        plot_state['fig'].canvas.flush_events()
