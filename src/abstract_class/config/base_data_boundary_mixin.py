from typing import Dict, Optional, Tuple

import numpy as np

from src.abstract_class.config.base_data_boundary_generation_mixin import (
    BaseDataBoundaryGenerationMixin,
)
from src.abstract_class.config.base_data_boundary_projection_mixin import (
    BaseDataBoundaryProjectionMixin,
)


class BaseDataBoundaryMixin(
    BaseDataBoundaryGenerationMixin,
    BaseDataBoundaryProjectionMixin,
):
    DEFAULT_BOUNDARY_SAMPLING_MODE = "global"

    def read_boundary_conditions(self) -> Dict:
        if not hasattr(self.config, "boundary_conditions") or not self.config.boundary_conditions:
            return {}
        boundary_data = self._initialize_boundary_data_store()
        for bc in self.config.boundary_conditions:
            self._append_boundary_condition(boundary_data, bc)
        return self._merge_boundary_data(boundary_data)

    def _append_boundary_condition(self, boundary_data: Dict, bc: Dict) -> None:
        bc_type = bc["type"].lower()
        points = bc["points"]
        sampling_mode = self._resolve_boundary_sampling_mode(bc)
        if bc_type == "periodic":
            self._append_periodic_boundary_condition(
                boundary_data,
                bc,
                points,
                sampling_mode,
            )
            return
        region = bc["region"]
        x_boundary = self._generate_training_boundary_points(
            region,
            points,
            sampling_mode,
        )
        if x_boundary.size == 0:
            return
        normals = None
        if bc_type != "dirichlet":
            normals = self._get_boundary_normals(region, x_boundary.shape[0])
        boundary_values = self._resolve_boundary_values(bc, x_boundary)
        boundary_weights = np.full((x_boundary.shape[0], 1), float(bc.get("weight", 1.0)))
        target_var_indices = self._resolve_boundary_target_indices(bc)
        self._append_boundary_values(
            boundary_data,
            bc_type,
            x_boundary,
            boundary_values,
            boundary_weights,
            normals,
            target_var_indices,
        )

    def _append_periodic_boundary_condition(
        self,
        boundary_data: Dict,
        bc: Dict,
        points: int,
        sampling_mode: str,
    ) -> None:
        regions = self._extract_periodic_regions(bc)
        if regions is None:
            return
        region, pair_region = regions
        x_boundary_1 = self._generate_training_boundary_points(
            region,
            points,
            sampling_mode,
        )
        x_boundary_2 = self._generate_training_boundary_points(
            pair_region,
            points,
            sampling_mode,
        )
        if x_boundary_1.size == 0 or x_boundary_2.size == 0:
            return
        self._append_periodic_boundary_pairs(
            boundary_data,
            bc,
            region,
            pair_region,
            x_boundary_1,
            x_boundary_2,
        )

    def _generate_training_boundary_points(
        self,
        region: str,
        points: int,
        sampling_mode: str,
    ) -> np.ndarray:
        if sampling_mode != "segment_local":
            return self._generate_boundary_points(region, points)

        boundary_parts = []
        for segment_idx in range(self.Ns):
            segment_boundary = self._generate_segment_boundary_points(
                region,
                points,
                segment_idx,
            )
            if segment_boundary.size > 0:
                boundary_parts.append(segment_boundary)

        if boundary_parts:
            return np.vstack(boundary_parts)
        return np.array([])

    def _resolve_boundary_sampling_mode(self, bc: Dict) -> str:
        if "sampling_mode" in bc:
            return bc["sampling_mode"]
        return getattr(
            self.config,
            "boundary_sampling_mode",
            self.DEFAULT_BOUNDARY_SAMPLING_MODE,
        )

    def _initialize_boundary_data_store(self) -> Dict:
        boundary_data = {}
        for var_idx in range(len(self.config.vars_list)):
            boundary_data[var_idx] = self._empty_boundary_entry()
        return boundary_data

    @staticmethod
    def _empty_boundary_entry() -> Dict:
        return {
            "dirichlet": {"x": [], "values": [], "weights": []},
            "neumann": {"x": [], "values": [], "weights": [], "normals": []},
            "robin": {"x": [], "values": [], "weights": [], "params": [], "normals": []},
            "periodic": {"pairs": []},
        }

    @staticmethod
    def _extract_periodic_regions(bc: Dict) -> Optional[Tuple[str, str]]:
        if "pairs" in bc:
            if len(bc["pairs"]) != 2:
                print(
                    "Warning: Periodic boundary pairs must contain exactly 2 regions, "
                    f"got {len(bc['pairs'])}"
                )
                return None
            return bc["pairs"][0], bc["pairs"][1]
        if "region" in bc and "pair_with" in bc:
            return bc["region"], bc["pair_with"]
        print("Warning: Invalid periodic boundary condition format - missing 'pairs' or 'region'+'pair_with'")
        return None

    def _append_periodic_boundary_pairs(
        self,
        boundary_data: Dict,
        bc: Dict,
        region: str,
        pair_region: str,
        x_boundary_1: np.ndarray,
        x_boundary_2: np.ndarray,
    ) -> None:
        periodic_pair = {
            "segment_1": self._region_to_segment(region),
            "segment_2": self._region_to_segment(pair_region),
            "x_1": x_boundary_1,
            "x_2": x_boundary_2,
            "weight": float(bc.get("weight", 1.0)),
        }
        target_var_indices = self._resolve_boundary_target_indices(bc)
        for var_idx in target_var_indices:
            boundary_data[var_idx]["periodic"]["pairs"].append(periodic_pair.copy())

    def _resolve_boundary_target_indices(self, bc: Dict) -> list:
        vars_list = list(getattr(self.config, "vars_list", []))
        if not vars_list:
            return []

        if "vars" in bc:
            selected = bc["vars"]
            if not isinstance(selected, list):
                raise ValueError(f"Boundary condition 'vars' must be a list: {bc}")
            return [vars_list.index(var_name) for var_name in selected]

        if "var" in bc:
            return [vars_list.index(bc["var"])]

        return list(range(len(vars_list)))

    def _resolve_boundary_values(self, bc: Dict, x_boundary: np.ndarray) -> np.ndarray:
        if "formula" in bc:
            return self._parse_math_expression(bc["formula"], x_boundary)
        if "value" in bc:
            value = bc["value"]
            if isinstance(value, str):
                return self._parse_math_expression(value, x_boundary)
            return np.full((x_boundary.shape[0], 1), float(value))
        raise ValueError(f"Boundary condition must have either 'value' or 'formula' field: {bc}")

    def _append_boundary_values(
        self,
        boundary_data: Dict,
        bc_type: str,
        x_boundary: np.ndarray,
        boundary_values: np.ndarray,
        boundary_weights: np.ndarray,
        normals: Optional[np.ndarray],
        target_var_indices: list,
    ) -> None:
        for var_idx in target_var_indices:
            boundary_data[var_idx][bc_type]["x"].append(x_boundary)
            boundary_data[var_idx][bc_type]["values"].append(boundary_values)
            boundary_data[var_idx][bc_type]["weights"].append(boundary_weights)
            if normals is not None:
                boundary_data[var_idx][bc_type]["normals"].append(normals)

    def _merge_boundary_data(self, boundary_data: Dict) -> Dict:
        for var_idx in boundary_data:
            for bc_type in boundary_data[var_idx]:
                if bc_type == "periodic":
                    continue
                self._merge_boundary_type(boundary_data[var_idx], bc_type)
        return boundary_data

    @staticmethod
    def _merge_boundary_type(boundary_entry: Dict, bc_type: str) -> None:
        if boundary_entry[bc_type]["x"]:
            boundary_entry[bc_type]["x"] = np.vstack(boundary_entry[bc_type]["x"])
            boundary_entry[bc_type]["values"] = np.vstack(boundary_entry[bc_type]["values"])
            boundary_entry[bc_type]["weights"] = np.vstack(boundary_entry[bc_type]["weights"])
            if bc_type in ["neumann", "robin"] and boundary_entry[bc_type]["normals"]:
                boundary_entry[bc_type]["normals"] = np.vstack(boundary_entry[bc_type]["normals"])
            return
        boundary_entry[bc_type]["x"] = np.array([])
        boundary_entry[bc_type]["values"] = np.array([])
        boundary_entry[bc_type]["weights"] = np.array([])
        if bc_type in ["neumann", "robin"]:
            boundary_entry[bc_type]["normals"] = np.array([])
