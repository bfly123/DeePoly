"""Lazy exports for visualization mixins."""

from importlib import import_module

_EXPORTS = {
    "ConvergenceMixin": ("src.abstract_class.config.visualize_mixins.convergence_mixin", "ConvergenceMixin"),
    "SolutionPlotMixin": ("src.abstract_class.config.visualize_mixins.solution_plot_mixin", "SolutionPlotMixin"),
    "ReferenceComparisonMixin": ("src.abstract_class.config.visualize_mixins.reference_comparison_mixin", "ReferenceComparisonMixin"),
    "GridReconstructionMixin": ("src.abstract_class.config.visualize_mixins.grid_reconstruction_mixin", "GridReconstructionMixin"),
    "LegacyAnalysisMixin": ("src.abstract_class.config.visualize_mixins.legacy_analysis_mixin", "LegacyAnalysisMixin"),
    "AnimationMixin": ("src.abstract_class.config.visualize_mixins.animation_mixin", "AnimationMixin"),
    "RealtimeVisualizationMixin": ("src.abstract_class.config.visualize_mixins.realtime_visualization_mixin", "RealtimeVisualizationMixin"),
    "ErrorAnalysisMixin": ("src.abstract_class.config.visualize_mixins.error_analysis_mixin", "ErrorAnalysisMixin"),
    "SegmentOverlayMixin": ("src.abstract_class.config.visualize_mixins.segment_overlay_mixin", "SegmentOverlayMixin"),
    "DatasetVisualizationMixin": ("src.abstract_class.config.visualize_mixins.dataset_visualization_mixin", "DatasetVisualizationMixin"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
