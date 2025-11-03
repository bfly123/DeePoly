import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Optional, Dict, Any, Union
from .base import PlotComponent, FigureContext
from .registry import PlotRegistry


class FigureBuilder:

    def __init__(self, figsize: Tuple[float, float] = (12, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.fig: Optional[plt.Figure] = None
        self.axes: List[plt.Axes] = []
        self.components: List[Tuple[PlotComponent, FigureContext]] = []

    def create_figure(self, nrows: int = 1, ncols: int = 1, **kwargs) -> plt.Figure:
        self.clear()
        self.fig, axes = plt.subplots(nrows, ncols, figsize=self.figsize, dpi=self.dpi, **kwargs)

        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else axes

        self.axes = axes if isinstance(axes, list) else list(axes)
        return self.fig

    def create_with_gridspec(self, grid_spec: Union[gridspec.GridSpecBase, List[Tuple[int, int, int, int]]]) -> plt.Figure:
        self.clear()
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)

        if isinstance(grid_spec, gridspec.GridSpecBase):
            for i in range(grid_spec.nrows * grid_spec.ncols):
                ax = self.fig.add_subplot(grid_spec[i])
                self.axes.append(ax)
        else:
            for spec in grid_spec:
                row, col, rowspan, colspan = spec
                ax = plt.subplot2grid((max(s[0] + s[2] for s in grid_spec),
                                      max(s[1] + s[3] for s in grid_spec)),
                                     (row, col), rowspan=rowspan, colspan=colspan)
                self.axes.append(ax)

        return self.fig

    def create_with_mosaic(self, mosaic: Union[str, List[List[str]]], **kwargs) -> plt.Figure:
        self.clear()
        self.fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=self.figsize, dpi=self.dpi, **kwargs)
        self.axes = list(ax_dict.values())
        return self.fig

    def add_component(self, component: PlotComponent, ax_index: int = 0, metadata: Optional[Dict[str, Any]] = None) -> 'FigureBuilder':
        if self.fig is None:
            raise RuntimeError("Create figure first using create_figure()")

        if ax_index >= len(self.axes):
            raise IndexError(f"Axis index {ax_index} out of range")

        context = FigureContext(
            fig=self.fig,
            ax=self.axes[ax_index],
            index=ax_index,
            metadata=metadata or {}
        )
        self.components.append((component, context))
        return self

    def add_component_by_name(self, name: str, ax_index: int = 0, metadata: Optional[Dict[str, Any]] = None, **config) -> 'FigureBuilder':
        component = PlotRegistry.create(name, **config)
        return self.add_component(component, ax_index, metadata)

    def render(self, data: Dict[str, Any]) -> plt.Figure:
        if not self.components:
            raise RuntimeError("No components added to builder")

        for component, context in self.components:
            component.render(context, data)

        plt.tight_layout()
        return self.fig

    def clear(self) -> None:
        self.components.clear()
        self.axes.clear()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
