from typing import Dict

import numpy as np
import torch


class BoundaryConditionGpuMixin:
    def prepare_boundary_gpu_data(self, data: Dict) -> Dict:
        if "global_boundary_dict" not in data:
            return {"global_boundary_dict": {}}
        global_boundary_dict = {}
        for var_idx, var_boundary in data["global_boundary_dict"].items():
            global_boundary_dict[var_idx] = self._prepare_var_boundary_gpu_entry(var_boundary)
        return {"global_boundary_dict": global_boundary_dict}

    def _prepare_var_boundary_gpu_entry(self, var_boundary: Dict) -> Dict:
        prepared = {}
        for bc_type, bc_data in var_boundary.items():
            prepared[bc_type] = (
                self._prepare_periodic_boundary_gpu_data(bc_data)
                if bc_type == "periodic"
                else self._prepare_standard_boundary_gpu_data(bc_data)
            )
        return prepared

    def _prepare_periodic_boundary_gpu_data(self, bc_data: Dict) -> Dict:
        prepared_pairs = []
        for pair in bc_data["pairs"]:
            prepared_pairs.append({key: self._to_boundary_tensor(key, value) for key, value in pair.items()})
        return {"pairs": prepared_pairs}

    def _prepare_standard_boundary_gpu_data(self, bc_data: Dict) -> Dict:
        prepared = {}
        for key, value in bc_data.items():
            prepared[key] = self._to_boundary_tensor(key, value)
        return prepared

    def _to_boundary_tensor(self, key: str, value):
        if isinstance(value, np.ndarray) and value.size > 0:
            requires_grad = key == "x" or key.startswith("x_")
            return torch.tensor(
                value,
                dtype=torch.float64,
                device=self.config.device,
                requires_grad=requires_grad,
            )
        return value
