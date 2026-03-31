from typing import Dict

import numpy as np


class BaseDataBoundarySegmentProcessingMixin:
    def _process_segment_boundary(self, global_boundary_data: Dict, segment_idx: int) -> Dict:
        if self._uses_segment_local_boundary_sampling():
            return self._process_segment_boundary_locally(global_boundary_data, segment_idx)
        segment_boundary_data = {}
        x_min, x_max = self.config.x_min, self.config.x_max
        for var_idx, global_var_entry in global_boundary_data.items():
            segment_boundary_data[var_idx] = self._empty_boundary_entry()
            self._project_var_boundary_data(
                segment_boundary_data[var_idx],
                global_var_entry,
                segment_idx,
                x_min,
                x_max,
            )
        return segment_boundary_data

    def _uses_segment_local_boundary_sampling(self) -> bool:
        legacy_mode = getattr(self.config, "boundary_sampling_mode", None)
        if legacy_mode == "segment_local":
            return True
        for bc in getattr(self.config, "boundary_conditions", []):
            if bc.get("sampling_mode") == "segment_local":
                return True
        return False

    def _process_segment_boundary_locally(
        self,
        global_boundary_data: Dict,
        segment_idx: int,
    ) -> Dict:
        segment_boundary_data = {
            var_idx: self._empty_boundary_entry()
            for var_idx in range(len(self.config.vars_list))
        }

        x_min, x_max = self.config.x_min, self.config.x_max
        for var_idx, global_var_entry in global_boundary_data.items():
            segment_boundary_data[var_idx]["periodic"]["pairs"] = self._filter_periodic_pairs_for_segment(
                global_var_entry.get("periodic", {}).get("pairs", []),
                segment_idx,
                x_min,
                x_max,
            )

        for bc in getattr(self.config, "boundary_conditions", []):
            bc_type = bc["type"].lower()
            if bc_type == "periodic":
                continue

            sampling_mode = bc.get("sampling_mode", "segment_local")
            if sampling_mode == "global":
                x_boundary = self._extract_global_boundary_points_for_segment(
                    bc["region"],
                    int(bc["points"]),
                    segment_idx,
                )
            else:
                x_boundary = self._generate_segment_boundary_points(
                    bc["region"],
                    int(bc["points"]),
                    segment_idx,
                )
            if x_boundary.size == 0:
                continue

            boundary_values = self._resolve_boundary_values(bc, x_boundary)
            boundary_weights = np.full((x_boundary.shape[0], 1), float(bc.get("weight", 1.0)))
            normals = None
            if bc_type != "dirichlet":
                normals = self._get_boundary_normals(bc["region"], x_boundary.shape[0])

            x_boundary_local = self._normalize_data(
                x_boundary,
                x_min[segment_idx],
                x_max[segment_idx],
            )
            for var_idx in self._resolve_boundary_target_indices(bc):
                entry = segment_boundary_data[var_idx][bc_type]
                entry["x"].append(x_boundary_local)
                entry["values"].append(boundary_values)
                entry["weights"].append(boundary_weights)
                if normals is not None:
                    entry["normals"].append(normals)
                if bc_type == "robin" and "params" in bc:
                    entry["params"] = bc["params"]

        for var_idx in segment_boundary_data:
            for bc_type in segment_boundary_data[var_idx]:
                if bc_type == "periodic":
                    continue
                self._merge_boundary_type(segment_boundary_data[var_idx], bc_type)
        return segment_boundary_data
