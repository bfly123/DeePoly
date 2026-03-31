from typing import Dict

import numpy as np
import torch


class BaseNetDataMixin:
    def _prepare_physics_loss_data(self, data_GPU: Dict, **kwargs) -> Dict:
        """Unified data extraction for physics_loss"""
        extracted = {
            "dt": kwargs.get("dt", 0),
            "step": kwargs.get("step", "pre"),
            "x_train": data_GPU.get("x_train", data_GPU.get("x", None)),
        }

        extracted["U"] = None
        if extracted["x_train"] is not None:
            _, extracted["U"] = self(extracted["x_train"])

        extracted["param"] = data_GPU.get("param", [{"Re": 1.0, "nu": 1.0}])
        if extracted["param"] and len(extracted["param"]) > 0:
            extracted["Re"] = extracted["param"][0].get("Re", 1.0)
            extracted["nu"] = extracted["param"][0].get("nu", 1.0)
        else:
            extracted["Re"] = 1.0
            extracted["nu"] = 1.0

        extracted["U_current"] = data_GPU.get("U_current", None)
        extracted["x_bd"] = data_GPU.get("x_bd", None)
        extracted["u_bd"] = data_GPU.get("u_bd", None)
        extracted["global_boundary_dict"] = data_GPU.get("global_boundary_dict", None)
        extracted["source"] = data_GPU.get("source", None)
        extracted["u_train"] = data_GPU.get("u_train", None)
        return extracted

    def compute_total_loss(
        self,
        pde_loss: torch.Tensor,
        data_GPU: Dict,
        boundary_weight: float = 10.0,
    ) -> torch.Tensor:
        """Unified total loss computation"""
        boundary_loss = self._compute_boundary_loss(data_GPU)
        return pde_loss + boundary_weight * boundary_loss

    def prepare_gpu_data(self, data: Dict, U_current: np.ndarray = None) -> Dict:
        """Unified GPU data preparation for all problem types"""
        gpu_data = {
            "x_train": torch.tensor(
                data["x"],
                dtype=torch.float64,
                device=self.config.device,
                requires_grad=True,
            )
        }
        if U_current is not None:
            gpu_data["U_current"] = torch.tensor(U_current, dtype=torch.float64, device=self.config.device)
        if "source" in data:
            gpu_data["source"] = torch.tensor(data["source"], dtype=torch.float64, device=self.config.device)

        gpu_data.update(self.prepare_boundary_gpu_data(data))
        self._extract_boundary_arrays(data, gpu_data)
        gpu_data["param"] = [{"Re": 1.0, "nu": 1.0}]
        return gpu_data

    def _extract_boundary_arrays(self, data: Dict, gpu_data: Dict) -> None:
        """Extract boundary coordinate and value arrays"""
        if "global_boundary_dict" not in data or not data["global_boundary_dict"]:
            self._set_empty_boundary_tensors(gpu_data)
            return

        x_bd_list = []
        u_bd_list = []
        for var_idx, boundary_types in data["global_boundary_dict"].items():
            for bc_data in boundary_types.values():
                x_bd = bc_data.get("x")
                u_bd = bc_data.get("values")
                if isinstance(x_bd, np.ndarray) and x_bd.size > 0:
                    x_bd_list.append(x_bd)
                    u_bd_list.append(u_bd)

        if x_bd_list:
            gpu_data["x_bd"] = torch.tensor(
                np.vstack(x_bd_list),
                dtype=torch.float64,
                device=self.config.device,
                requires_grad=True,
            )
            gpu_data["u_bd"] = torch.tensor(
                np.vstack(u_bd_list),
                dtype=torch.float64,
                device=self.config.device,
            )
            return

        self._set_empty_boundary_tensors(gpu_data)

    def _set_empty_boundary_tensors(self, gpu_data: Dict) -> None:
        gpu_data["x_bd"] = torch.zeros((0, self.config.n_dim), dtype=torch.float64, device=self.config.device)
        gpu_data["u_bd"] = torch.zeros((0, self.config.n_eqs), dtype=torch.float64, device=self.config.device)

    def prepare_train_data(self, data: Dict) -> Dict:
        """Convert data to tensors with unified processing"""
        return {
            key: torch.tensor(value, dtype=torch.float64, device=self.config.device)
            for key, value in data.items()
            if value is not None
        }
