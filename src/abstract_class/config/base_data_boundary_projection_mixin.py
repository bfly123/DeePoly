from .base_data_boundary_region_mixin import BaseDataBoundaryRegionMixin
from .base_data_boundary_segment_points_mixin import BaseDataBoundarySegmentPointMixin
from .base_data_boundary_segment_processing_mixin import (
    BaseDataBoundarySegmentProcessingMixin,
)
from .base_data_boundary_segment_projection_mixin import (
    BaseDataBoundarySegmentProjectionMixin,
)


class BaseDataBoundaryProjectionMixin(
    BaseDataBoundarySegmentProcessingMixin,
    BaseDataBoundarySegmentProjectionMixin,
    BaseDataBoundarySegmentPointMixin,
    BaseDataBoundaryRegionMixin,
):
    """Facade for segment-aware boundary projection helpers."""
