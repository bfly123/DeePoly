from typing import Dict, Tuple

import torch


class BoundaryConditionLossMixin:
    def compute_boundary_loss(self, data_GPU: Dict, network) -> torch.Tensor:
        global_boundary_dict = data_GPU.get("global_boundary_dict", {})
        bc_handlers = {
            "dirichlet": self._compute_dirichlet_loss,
            "neumann": self._compute_neumann_loss,
            "robin": self._compute_robin_loss,
            "periodic": self._compute_periodic_loss,
        }
        total_error = None
        total_weight = 0.0
        for var_idx, var_bc_dict in global_boundary_dict.items():
            for bc_type, handler in bc_handlers.items():
                if bc_type in var_bc_dict and self._has_valid_bc_data(var_bc_dict[bc_type]):
                    error_sum, error_weight = handler(var_bc_dict[bc_type], network, var_idx)
                    if error_weight <= 0:
                        continue
                    if total_error is None:
                        total_error = error_sum
                    else:
                        total_error = total_error + error_sum
                    total_weight += error_weight

        if total_weight <= 0:
            return torch.tensor(0.0, dtype=torch.float64, device=network.config.device)
        return total_error / total_weight

    def _has_valid_bc_data(self, bc_data: Dict) -> bool:
        if not bc_data:
            return False
        if "x" in bc_data:
            return bc_data["x"].shape[0] > 0
        if "pairs" in bc_data:
            return bool(bc_data["pairs"])
        return False

    def _compute_dirichlet_loss(self, bc_data: Dict, network, var_idx: int) -> Tuple[torch.Tensor, int]:
        x_bc = bc_data["x"]
        u_bc = bc_data["values"]
        _, pred_bc = network(x_bc)
        weights = bc_data.get("weights")
        if weights is None or getattr(weights, "numel", lambda: 0)() == 0:
            weights = torch.ones_like(u_bc)
        errors = weights * (pred_bc[:, var_idx : var_idx + 1] - u_bc) ** 2
        return torch.sum(errors), float(torch.sum(weights).detach().item())

    def _compute_neumann_loss(self, bc_data: Dict, network, var_idx: int) -> Tuple[torch.Tensor, int]:
        return self._compute_gradient_boundary_loss(
            bc_data,
            network,
            var_idx,
            robin_params=None,
        )

    def _compute_robin_loss(self, bc_data: Dict, network, var_idx: int) -> Tuple[torch.Tensor, int]:
        return self._compute_gradient_boundary_loss(
            bc_data,
            network,
            var_idx,
            robin_params=bc_data["params"],
        )

    def _compute_gradient_boundary_loss(self, bc_data: Dict, network, var_idx: int, robin_params) -> Tuple[torch.Tensor, int]:
        x_bc = bc_data["x"]
        u_bc = bc_data["values"]
        normals = bc_data["normals"]
        errors = []
        weights = bc_data.get("weights")
        for index in range(x_bc.shape[0]):
            robin_value = self._evaluate_boundary_point(
                network,
                x_bc[index : index + 1],
                normals[index],
                var_idx,
                robin_params,
            )
            point_weight = 1.0
            if weights is not None and getattr(weights, "numel", lambda: 0)() > 0:
                point_weight = weights[index]
            errors.append(point_weight * (robin_value - u_bc[index]) ** 2)
        if errors:
            stacked = torch.stack(errors)
            if weights is not None and getattr(weights, "numel", lambda: 0)() > 0:
                total_weight = float(torch.sum(weights).detach().item())
            else:
                total_weight = float(len(errors))
            return torch.sum(stacked), total_weight
        return torch.tensor(0.0, dtype=torch.float64, device=network.config.device), 0.0

    def _evaluate_boundary_point(self, network, x_point, normal, var_idx: int, robin_params):
        x_point = x_point.clone().detach().requires_grad_(True)
        _, u_pred = network(x_point)
        u_component = u_pred[:, var_idx : var_idx + 1]
        if robin_params is None:
            grads = network.gradients(u_component, x_point)[0]
            return torch.sum(grads * normal)
        alpha, beta = robin_params[0], robin_params[1]
        if abs(beta) <= 1e-10:
            return alpha * u_component
        grads = network.gradients(u_component, x_point)[0]
        normal_derivative = torch.sum(grads * normal)
        return alpha * u_component + beta * normal_derivative

    def _compute_periodic_loss(self, periodic_data: Dict, network, var_idx: int) -> Tuple[torch.Tensor, int]:
        total_error = None
        total_weight = 0.0
        for pair in periodic_data["pairs"]:
            _, pred_bc_1 = network(pair["x_1"])
            _, pred_bc_2 = network(pair["x_2"])
            pair_weight = float(pair.get("weight", 1.0))
            errors = pair_weight * (pred_bc_1[:, var_idx : var_idx + 1] - pred_bc_2[:, var_idx : var_idx + 1]) ** 2
            if total_error is None:
                total_error = torch.sum(errors)
            else:
                total_error = total_error + torch.sum(errors)
            total_weight += pair_weight * int(errors.numel())
        if total_weight <= 0:
            return torch.tensor(0.0, dtype=torch.float64, device=network.config.device), 0.0
        return total_error, total_weight
