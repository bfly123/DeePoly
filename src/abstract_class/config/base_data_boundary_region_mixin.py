import numpy as np


class BaseDataBoundaryRegionMixin:
    def _region_to_segment(self, region: str) -> int:
        dim_region = self._parse_dim_region(region)
        if dim_region is not None:
            return self._segment_index_for_dim_region(dim_region)
        if self.n_dim == 1:
            return {"left": 0, "right": self.n_segments[0] - 1}.get(region, 0)
        if self.n_dim == 2:
            return {
                "left": 0,
                "right": self.n_segments[0] - 1,
                "bottom": 0,
                "top": (self.n_segments[1] - 1) * self.n_segments[0],
            }.get(region, 0)
        return 0

    def _segment_index_for_dim_region(self, dim_region) -> int:
        dim_idx, is_max = dim_region
        segment_coords = [0] * self.n_dim
        segment_coords[dim_idx] = self.n_segments[dim_idx] - 1 if is_max else 0
        return self._coords_to_segment_index(segment_coords)

    def _coords_to_segment_index(self, coords: list) -> int:
        idx = 0
        for index, coord in enumerate(coords):
            idx += coord if index == 0 else coord * np.prod(self.n_segments[:index])
        return idx
