import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc


@dataclass
class BlockTrainingData:
    y_min: float
    y_max: float
    train_data: Dict
    interface_points: Optional[np.ndarray]
    interface_targets: Optional[np.ndarray]
    history_points: Optional[np.ndarray]
    history_targets: Optional[np.ndarray]
    guidance_points: Optional[np.ndarray]
    guidance_targets: Optional[np.ndarray]


class BCPINNInitializer:
    """Sequential bc-PINN style spotter for 2D space-time steady formulations.

    The implementation keeps the existing PyTorch model and boundary-condition
    handling, but trains the spotter sequentially over blocks of the selected
    time-like axis. For Allen-Cahn in ``New_test_2D``, ``y`` is treated as the
    physical time axis.
    """

    def __init__(self, solver):
        self.solver = solver
        self.config = solver.config
        self.model = solver.model
        self.data_train = solver.data_train
        self.device = self.config.device
        self.dtype = torch.float64
        self.tol = 1e-12

        init_cfg = getattr(self.config, "initial_condition", {}) or {}
        raw_cfg = init_cfg.get("bc_pinn_config", {}) if isinstance(init_cfg, dict) else {}

        self.block_axis = self._resolve_block_axis(raw_cfg.get("block_axis", "y"))
        default_blocks = self._default_block_count()
        self.n_time_blocks = int(raw_cfg.get("n_time_blocks", default_blocks))
        self.epochs_per_block = int(raw_cfg.get("epochs_per_block", self.config.epochs_adam))
        self.learning_rate = float(raw_cfg.get("learning_rate", 1e-3))
        self.warmup_learning_rate = float(raw_cfg.get("warmup_learning_rate", 1e-2))
        self.interface_weight = float(raw_cfg.get("interface_weight", 64.0))
        self.history_weight = float(raw_cfg.get("history_weight", 64.0))
        self.guidance_weight = float(raw_cfg.get("guidance_weight", 1.0))
        self.warmup_ratio = float(raw_cfg.get("warmup_ratio", 0.1))
        self.freeze_hidden_layers = int(raw_cfg.get("freeze_hidden_layers", 2))
        self.interface_points = int(raw_cfg.get("interface_points", 128))
        self.history_points = int(raw_cfg.get("history_points", 0))
        self.guidance_points = int(raw_cfg.get("guidance_points", 0))
        self.collocation_points = int(raw_cfg.get("collocation_points", 20000))
        self.x_line_points = int(raw_cfg.get("x_line_points", 128))
        self.total_time_points = int(raw_cfg.get("total_time_points", 201))
        self.block_time_points = int(raw_cfg.get("block_time_points", 51))
        self.boundary_time_points = int(raw_cfg.get("boundary_time_points", max(1, self.block_time_points // 2)))
        self.lbfgs_steps = int(raw_cfg.get("lbfgs_steps", 0))
        self.history_mode = str(raw_cfg.get("history_mode", "previous_only"))
        self.print_interval = int(raw_cfg.get("print_interval", 500))
        self.target_loss = float(raw_cfg.get("target_loss", self.config.DNNtol))
        self.seed = int(raw_cfg.get("seed", self.config.seed))

        self.time_min = float(self.config.x_domain[self.block_axis][0])
        self.time_max = float(self.config.x_domain[self.block_axis][1])
        self.block_bounds = np.linspace(self.time_min, self.time_max, self.n_time_blocks + 1)
        self.time_grid = np.linspace(self.time_min, self.time_max, self.total_time_points)
        self.time_index_bounds = np.linspace(0, self.total_time_points - 1, self.n_time_blocks + 1, dtype=int)

    def run(self) -> np.ndarray:
        print("  Sequential bc-PINN spotter settings:")
        print(f"    time axis: {self.config.spatial_vars[self.block_axis]} (dim={self.block_axis})")
        print(f"    blocks: {self.n_time_blocks}")
        print(f"    epochs/block: {self.epochs_per_block}")
        print(f"    interface/history/guidance weights: {self.interface_weight}/{self.history_weight}/{self.guidance_weight}")
        print(f"    freeze_hidden_layers: {self.freeze_hidden_layers}")

        diagnostics = []
        for block_idx in range(self.n_time_blocks):
            block = self._build_block_training_data(block_idx)
            if block.train_data["x"].shape[0] == 0:
                print(f"  Block {block_idx + 1}/{self.n_time_blocks}: skipped (no train points)")
                continue
            stats = self._train_block(block_idx, block)
            diagnostics.append(stats)

        self._set_linear_layers_trainable(self.freeze_hidden_layers, False)
        self.model.eval()
        U_init = self.solver._predict_current_train_field()
        self._save_diagnostics(diagnostics)
        return U_init

    def _resolve_block_axis(self, axis_value) -> int:
        if isinstance(axis_value, int):
            return axis_value
        if isinstance(axis_value, str):
            spatial_vars = list(getattr(self.config, "spatial_vars", []))
            if axis_value in spatial_vars:
                return spatial_vars.index(axis_value)
            lowered = axis_value.lower()
            if lowered in spatial_vars:
                return spatial_vars.index(lowered)
            if lowered == "t" and "y" in spatial_vars:
                return spatial_vars.index("y")
        if self.config.n_dim == 2:
            return 1
        return 0

    def _default_block_count(self) -> int:
        n_segments = list(getattr(self.config, "n_segments", []))
        if len(n_segments) > self.block_axis:
            return max(1, int(n_segments[self.block_axis]))
        return 4

    def _build_block_training_data(self, block_idx: int) -> BlockTrainingData:
        y_min = float(self.block_bounds[block_idx])
        y_max = float(self.block_bounds[block_idx + 1])
        is_first_block = block_idx == 0
        block_time_grid = self._get_block_time_grid(block_idx)
        history_time_grid = self._get_history_time_grid(block_idx)
        collocation = self._sample_block_collocation(y_min, y_max, self.collocation_points)
        train_data = {
            "x": collocation,
            "global_boundary_dict": self._build_periodic_boundary_dict(block_time_grid),
        }
        train_data["source"] = self.solver.datagen._load_source_term(collocation)

        interface_points, interface_targets = self._build_interface_targets(y_min, is_first_block)
        history_points, history_targets = self._build_history_targets(history_time_grid, block_idx)
        guidance_points, guidance_targets = self._build_guidance_targets(block_time_grid, y_min, is_first_block)

        return BlockTrainingData(
            y_min=y_min,
            y_max=y_max,
            train_data=train_data,
            interface_points=interface_points,
            interface_targets=interface_targets,
            history_points=history_points,
            history_targets=history_targets,
            guidance_points=guidance_points,
            guidance_targets=guidance_targets,
        )

    def _get_block_time_grid(self, block_idx: int) -> np.ndarray:
        idx_start = int(self.time_index_bounds[block_idx])
        idx_end = int(self.time_index_bounds[block_idx + 1])
        if self.block_time_points > 0:
            return np.linspace(self.time_grid[idx_start], self.time_grid[idx_end], self.block_time_points)
        return self.time_grid[idx_start: idx_end + 1]

    def _get_history_time_grid(self, block_idx: int) -> np.ndarray:
        if self.history_mode == "cumulative_current":
            idx_end = int(self.time_index_bounds[block_idx + 1])
            return self.time_grid[: idx_end + 1]
        idx_end = int(self.time_index_bounds[block_idx])
        return self.time_grid[: idx_end + 1]

    def _sample_block_collocation(self, y_min: float, y_max: float, n_points: int) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=2, seed=self.seed)
        samples = sampler.random(n_points)
        x_min = float(self.config.x_domain[0][0])
        x_max = float(self.config.x_domain[0][1])
        lower = np.array([x_min, y_min], dtype=np.float64)
        upper = np.array([x_max, y_max], dtype=np.float64)
        if self.block_axis == 0:
            lower = np.array([y_min, x_min], dtype=np.float64)
            upper = np.array([y_max, x_max], dtype=np.float64)
        return qmc.scale(samples, lower, upper)

    def _build_periodic_boundary_dict(self, block_time_grid: np.ndarray) -> Dict:
        if self.config.n_dim != 2:
            return {}
        rng = np.random.default_rng(self.seed)
        sample_count = min(len(block_time_grid), self.boundary_time_points)
        time_samples = np.sort(rng.choice(block_time_grid, sample_count, replace=False))
        x_left, x_right = self._build_periodic_boundary_pairs(time_samples)
        return {
            0: {
                "dirichlet": {"x": np.array([]), "values": np.array([]), "weights": np.array([])},
                "neumann": {"x": np.array([]), "values": np.array([]), "weights": np.array([]), "normals": np.array([])},
                "robin": {"x": np.array([]), "values": np.array([]), "weights": np.array([]), "params": [], "normals": np.array([])},
                "periodic": {
                    "pairs": [
                        {
                            "segment_1": 0,
                            "segment_2": 0,
                            "x_1": x_left,
                            "x_2": x_right,
                            "weight": 1.0,
                        }
                    ]
                },
            }
        }

    def _build_periodic_boundary_pairs(self, time_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left = np.zeros((len(time_samples), 2), dtype=np.float64)
        right = np.zeros((len(time_samples), 2), dtype=np.float64)
        x_min = float(self.config.x_domain[0][0])
        x_max = float(self.config.x_domain[0][1])
        if self.block_axis == 1:
            left[:, 0] = x_min
            left[:, 1] = time_samples
            right[:, 0] = x_max
            right[:, 1] = time_samples
        else:
            left[:, 0] = time_samples
            left[:, 1] = x_min
            right[:, 0] = time_samples
            right[:, 1] = x_max
        return left, right

    def _build_interface_targets(self, y_start: float, is_first_block: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x_line = self._sample_interface_line(y_start, self.interface_points)
        if x_line.size == 0:
            return None, None
        if is_first_block:
            return x_line, self._evaluate_initial_profile(x_line)
        return x_line, self._predict_numpy(x_line)

    def _build_history_targets(self, cumulative_time_grid: np.ndarray, block_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if block_idx == 0:
            return None, None
        history_points = self._build_space_time_mesh(cumulative_time_grid)
        if self.history_points > 0:
            history_points = self._sample_rows(history_points, self.history_points)
        if history_points is None or history_points.size == 0:
            return None, None
        return history_points, self._predict_numpy(history_points)

    def _build_guidance_targets(
        self,
        block_time_grid: np.ndarray,
        y_start: float,
        is_first_block: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        sampled_points = self._build_space_time_mesh(block_time_grid)
        if self.guidance_points > 0:
            sampled_points = self._sample_rows(sampled_points, self.guidance_points)
        if sampled_points is None or sampled_points.size == 0:
            return None, None
        projected = sampled_points.copy()
        projected[:, self.block_axis] = y_start
        if is_first_block:
            targets = self._evaluate_initial_profile(projected)
        else:
            targets = self._predict_numpy(projected)
        return sampled_points, targets

    def _sample_interface_line(self, y_value: float, n_points: int) -> np.ndarray:
        if self.config.n_dim != 2 or n_points <= 0:
            return np.array([])
        x_min = float(self.config.x_domain[0][0])
        x_max = float(self.config.x_domain[0][1])
        x_values = np.linspace(x_min, x_max, n_points)
        y_values = np.full_like(x_values, y_value)
        if self.block_axis == 1:
            return np.column_stack([x_values, y_values])
        return np.column_stack([y_values, x_values])

    def _build_space_time_mesh(self, time_grid: np.ndarray) -> np.ndarray:
        x_line = self._sample_interface_line(0.0, self.x_line_points)
        if x_line.size == 0:
            return np.array([])
        if self.block_axis == 1:
            x_values = x_line[:, 0]
            X, T = np.meshgrid(x_values, time_grid)
            return np.column_stack([X.reshape(-1), T.reshape(-1)])
        y_values = x_line[:, 1]
        T, Y = np.meshgrid(time_grid, y_values)
        return np.column_stack([T.reshape(-1), Y.reshape(-1)])

    def _evaluate_initial_profile(self, points: np.ndarray) -> np.ndarray:
        bottom_bc = self._find_bottom_boundary_condition()
        if bottom_bc is None:
            return np.zeros((points.shape[0], self.config.n_eqs), dtype=np.float64)
        if "formula" in bottom_bc:
            values = self.solver.datagen._parse_math_expression(bottom_bc["formula"], points)
            values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        elif "value" in bottom_bc:
            raw_value = bottom_bc["value"]
            if isinstance(raw_value, str):
                values = self.solver.datagen._parse_math_expression(raw_value, points)
                values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
            else:
                values = np.full((points.shape[0], 1), float(raw_value), dtype=np.float64)
        else:
            values = np.zeros((points.shape[0], 1), dtype=np.float64)
        return values

    def _find_bottom_boundary_condition(self) -> Optional[Dict]:
        for bc in getattr(self.config, "boundary_conditions", []):
            if bc.get("type", "").lower() == "dirichlet" and bc.get("region") == "bottom":
                return bc
        return None

    def _predict_numpy(self, points: np.ndarray) -> np.ndarray:
        if points is None or points.size == 0:
            return np.zeros((0, self.config.n_eqs), dtype=np.float64)
        x_tensor = torch.tensor(points, dtype=self.dtype, device=self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_tensor)
        pred = output[-1] if isinstance(output, tuple) else output
        return pred.detach().cpu().numpy()

    def _sample_rows(self, array: np.ndarray, sample_count: int) -> Optional[np.ndarray]:
        if array is None or array.size == 0:
            return None
        if array.shape[0] <= sample_count:
            return np.asarray(array, dtype=np.float64)
        rng = np.random.default_rng(self.seed + array.shape[0])
        indices = rng.choice(array.shape[0], sample_count, replace=False)
        return np.asarray(array[indices], dtype=np.float64)

    def _train_block(self, block_idx: int, block: BlockTrainingData) -> Dict:
        start_time = time.time()
        self.model.train()
        self._set_linear_layers_trainable(self.freeze_hidden_layers, block_idx > 0)
        optimizer = optim.Adam(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.warmup_learning_rate,
        )
        data_gpu = self.model.prepare_gpu_data(block.train_data)
        warmup_epochs = int(self.epochs_per_block * self.warmup_ratio)

        interface_x, interface_u = self._to_tensor_pair(block.interface_points, block.interface_targets)
        history_x, history_u = self._to_tensor_pair(block.history_points, block.history_targets)
        guidance_x, guidance_u = self._to_tensor_pair(block.guidance_points, block.guidance_targets)

        best_state = None
        best_loss = float("inf")
        final_stats = {}
        best_metric_name = "warmup_guidance" if warmup_epochs > 0 else "main_objective"

        print(
            f"  Block {block_idx + 1}/{self.n_time_blocks}: "
            f"{self.config.spatial_vars[self.block_axis]} in [{block.y_min:.6f}, {block.y_max:.6f}] "
            f"with {block.train_data['x'].shape[0]} train points"
        )

        for epoch in range(self.epochs_per_block):
            optimizer.zero_grad()
            pde_loss = self.model.physics_loss(data_gpu)
            interface_loss = self._tensor_mse(interface_x, interface_u)
            history_loss = self._tensor_mse(history_x, history_u)
            guidance_loss = self._tensor_mse(guidance_x, guidance_u)

            if epoch < warmup_epochs:
                total_loss = self.guidance_weight * guidance_loss
            else:
                if epoch == warmup_epochs:
                    for group in optimizer.param_groups:
                        group["lr"] = self.learning_rate
                total_loss = pde_loss
                total_loss = total_loss + self.interface_weight * interface_loss
                total_loss = total_loss + self.history_weight * history_loss

            total_loss.backward()
            optimizer.step()

            current_loss = float(total_loss.detach().item())
            track_for_best = epoch >= warmup_epochs or warmup_epochs == 0
            if track_for_best and current_loss < best_loss:
                best_loss = current_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_metric_name = "main_objective"

            if epoch % self.print_interval == 0 or epoch == self.epochs_per_block - 1:
                final_stats = {
                    "epoch": epoch,
                    "total_loss": current_loss,
                    "pde_loss": float(pde_loss.detach().item()),
                    "interface_loss": float(interface_loss.detach().item()),
                    "history_loss": float(history_loss.detach().item()),
                    "guidance_loss": float(guidance_loss.detach().item()),
                }
                print(
                    f"    Epoch {epoch:6d} | total={final_stats['total_loss']:.6e} | "
                    f"pde={final_stats['pde_loss']:.6e} | "
                    f"interface={final_stats['interface_loss']:.6e} | "
                    f"history={final_stats['history_loss']:.6e} | "
                    f"guide={final_stats['guidance_loss']:.6e}"
                )

            if current_loss < self.target_loss and epoch >= warmup_epochs:
                break

        if self.lbfgs_steps > 0:
            optimizer_lbfgs = optim.LBFGS(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=0.5,
                max_iter=self.lbfgs_steps,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer_lbfgs.zero_grad()
                pde_loss = self.model.physics_loss(data_gpu)
                interface_loss = self._tensor_mse(interface_x, interface_u)
                history_loss = self._tensor_mse(history_x, history_u)
                loss = pde_loss + self.interface_weight * interface_loss + self.history_weight * history_loss
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)
            final_lbfgs_loss = float(closure().detach().item())
            if final_lbfgs_loss < best_loss:
                best_loss = final_lbfgs_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_metric_name = "lbfgs_objective"

        if best_state is None:
            best_loss = float(final_stats.get("total_loss", 0.0))
            best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            best_metric_name = "final_state"

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        return {
            "block_index": block_idx,
            "y_min": block.y_min,
            "y_max": block.y_max,
            "train_points": int(block.train_data["x"].shape[0]),
            "interface_points": 0 if block.interface_points is None else int(block.interface_points.shape[0]),
            "history_points": 0 if block.history_points is None else int(block.history_points.shape[0]),
            "guidance_points": 0 if block.guidance_points is None else int(block.guidance_points.shape[0]),
            "best_loss": best_loss,
            "best_metric": best_metric_name,
            "elapsed_time": time.time() - start_time,
            "final_stats": final_stats,
        }

    def _to_tensor_pair(
        self,
        points: Optional[np.ndarray],
        values: Optional[np.ndarray],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if points is None or values is None:
            return None, None
        x_tensor = torch.tensor(points, dtype=self.dtype, device=self.device)
        u_tensor = torch.tensor(values, dtype=self.dtype, device=self.device)
        return x_tensor, u_tensor

    def _tensor_mse(self, x_tensor: Optional[torch.Tensor], target: Optional[torch.Tensor]) -> torch.Tensor:
        if x_tensor is None or target is None:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        output = self.model(x_tensor)
        pred = output[-1] if isinstance(output, tuple) else output
        return torch.mean((pred - target) ** 2)

    def _set_linear_layers_trainable(self, freeze_count: int, enable_freeze: bool) -> None:
        linear_layers: List[nn.Linear] = [
            layer for layer in self.model.net if isinstance(layer, nn.Linear)
        ]
        for idx, layer in enumerate(linear_layers):
            trainable = not (enable_freeze and idx < freeze_count)
            for param in layer.parameters():
                param.requires_grad = trainable
        for param in self.model.out.parameters():
            param.requires_grad = True

    def _save_diagnostics(self, diagnostics: List[Dict]) -> None:
        results_dir = getattr(self.config, "results_dir", None)
        if not results_dir:
            return
        os.makedirs(results_dir, exist_ok=True)
        payload = {
            "method": "bc_pinn",
            "time_axis": self.config.spatial_vars[self.block_axis],
            "n_time_blocks": self.n_time_blocks,
            "block_bounds": self.block_bounds.tolist(),
            "epochs_per_block": self.epochs_per_block,
            "learning_rate": self.learning_rate,
            "warmup_learning_rate": self.warmup_learning_rate,
            "collocation_points": self.collocation_points,
            "x_line_points": self.x_line_points,
            "total_time_points": self.total_time_points,
            "block_time_points": self.block_time_points,
            "boundary_time_points": self.boundary_time_points,
            "interface_weight": self.interface_weight,
            "history_weight": self.history_weight,
            "guidance_weight": self.guidance_weight,
            "warmup_ratio": self.warmup_ratio,
            "freeze_hidden_layers": self.freeze_hidden_layers,
            "lbfgs_steps": self.lbfgs_steps,
            "history_mode": self.history_mode,
            "diagnostics": diagnostics,
        }
        path = os.path.join(results_dir, "bc_pinn_diagnostics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
