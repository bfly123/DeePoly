from typing import Dict

import numpy as np


class BaseDataBoundarySegmentProjectionMixin:
    def _project_var_boundary_data(
        self,
        segment_entry: Dict,
        global_entry: Dict,
        segment_idx: int,
        x_min,
        x_max,
    ) -> None:
        for bc_type, bc_entry in global_entry.items():
            if bc_type == "periodic":
                segment_entry[bc_type]["pairs"] = self._filter_periodic_pairs_for_segment(
                    bc_entry["pairs"],
                    segment_idx,
                    x_min,
                    x_max,
                )
                continue
            if "x" not in bc_entry or len(bc_entry["x"]) == 0:
                continue
            self._project_standard_boundary_to_segment(
                segment_entry,
                global_entry,
                bc_type,
                segment_idx,
                x_min,
                x_max,
            )

    def _filter_periodic_pairs_for_segment(self, pairs, segment_idx: int, x_min, x_max):
        segment_pairs = []
        for pair in pairs:
            segment_pair = self._project_periodic_pair(pair, segment_idx, x_min, x_max)
            if segment_pair is not None:
                segment_pairs.append(segment_pair)
        return segment_pairs

    def _project_periodic_pair(self, pair: Dict, segment_idx: int, x_min, x_max):
        x_boundary_1 = pair["x_1"]
        x_boundary_2 = pair["x_2"]
        mask_1 = self._create_segment_mask(x_boundary_1, segment_idx)
        mask_2 = self._create_segment_mask(x_boundary_2, segment_idx)
        if not np.any(mask_1) and not np.any(mask_2):
            return None
        segment_pair = pair.copy()
        if np.any(mask_1):
            segment_pair["x_1"] = self._normalize_data(
                x_boundary_1[mask_1],
                x_min[segment_idx],
                x_max[segment_idx],
            )
        if np.any(mask_2):
            segment_pair["x_2"] = self._normalize_data(
                x_boundary_2[mask_2],
                x_min[segment_idx],
                x_max[segment_idx],
            )
        return segment_pair

    def _project_standard_boundary_to_segment(
        self,
        segment_entry: Dict,
        global_entry: Dict,
        bc_type: str,
        segment_idx: int,
        x_min,
        x_max,
    ) -> None:
        x_boundary = global_entry[bc_type]["x"]
        mask = self._create_segment_mask(x_boundary, segment_idx)
        if not np.any(mask):
            return
        segment_entry[bc_type]["x"] = self._normalize_data(
            x_boundary[mask],
            x_min[segment_idx],
            x_max[segment_idx],
        )
        segment_entry[bc_type]["values"] = global_entry[bc_type]["values"][mask]
        if bc_type in ["neumann", "robin"] and "normals" in global_entry[bc_type]:
            segment_entry[bc_type]["normals"] = global_entry[bc_type]["normals"][mask]
        if bc_type == "robin" and "params" in global_entry[bc_type]:
            segment_entry[bc_type]["params"] = global_entry[bc_type]["params"]
