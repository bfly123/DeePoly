from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.abstract_class.spotter_feature_provider import PaddleSpotterFeatureProvider
from src.spotter_runtime.backends.pde_program_common import build_symbolic_pde_context


_TOKEN_PATTERN = re.compile(r"\b[A-Za-z_]\w*\b")


class _PaddleSymbolicPDENet:
    def __init__(self, paddle, n_dim: int, hidden_dims, n_eqs: int, *, seed: int):
        import paddle.nn as nn

        paddle.seed(int(seed))
        dims = [int(n_dim)] + list(hidden_dims)
        layers = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(fan_in, fan_out))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], int(n_eqs), bias_attr=False)

    def parameters(self):
        return list(self.net.parameters()) + list(self.out.parameters())

    def state_dict(self):
        state = {}
        state.update({f"net.{key}": value.clone() for key, value in self.net.state_dict().items()})
        state.update({f"out.{key}": value.clone() for key, value in self.out.state_dict().items()})
        return state

    def set_state_dict(self, state):
        net_state = {key.split("net.", 1)[1]: value for key, value in state.items() if key.startswith("net.")}
        out_state = {key.split("out.", 1)[1]: value for key, value in state.items() if key.startswith("out.")}
        self.net.set_state_dict(net_state)
        self.out.set_state_dict(out_state)

    def train(self):
        self.net.train()
        self.out.train()

    def eval(self):
        self.net.eval()
        self.out.eval()

    def __call__(self, x):
        hidden = self.net(x)
        output = self.out(hidden)
        return hidden, output


class TrainablePaddleSymbolicPDEProvider(PaddleSpotterFeatureProvider):
    backend_name = "paddle"

    def __init__(self, *, config=None, data_train: Optional[Dict[str, Any]] = None, program: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        import paddle

        self.paddle = paddle
        self.context = dict(program or build_symbolic_pde_context(config, data_train))
        if device is not None:
            paddle.set_device(str(device))
        self.model = self._create_model(int(self.context["seed"]))
        self._x_train = np.asarray(self.context["x_train"], dtype=np.float64)
        self._boundary_program = self.context["boundary_program"]
        self._term_evaluators = self._build_term_evaluators()

        super().__init__(
            hidden_batch_fn=self._hidden_batch_fn,
            output_batch_fn=self._output_batch_fn,
            device=device,
            feature_dim=int(self.context["hidden_dims"][-1]),
            dtype="float64",
        )

    def supports_external_training(self) -> bool:
        return True

    def train_external(self, request: Dict[str, Any]) -> Dict[str, Any]:
        problem_type = str(request.get("problem_type") or self.context.get("problem_type") or "").lower()
        if problem_type == "time_pde" and "dt" in request and "U_current" in request:
            return self.train_external_step(
                problem_type=problem_type,
                dt=float(request["dt"]),
                U_current=np.asarray(request["U_current"], dtype=np.float64),
                iteration=request.get("iteration"),
            )
        if problem_type != "linear_pde":
            raise ValueError(f"Paddle symbolic PDE provider does not support train_external for problem_type={problem_type}")
        self._ensure_context_problem_type("linear_pde")
        return self._train_stationary()

    def train_external_step(self, *, problem_type: str, dt: float, U_current: np.ndarray, iteration: Optional[int] = None) -> Dict[str, Any]:
        _ = iteration
        requested = str(problem_type or "").lower()
        if requested != "time_pde":
            raise ValueError(f"Paddle symbolic PDE provider only supports train_external_step for time_pde. Got: {problem_type}")
        self._ensure_context_problem_type("time_pde")
        return self._train_time_step(dt=float(dt), U_current=np.asarray(U_current, dtype=np.float64))

    def _ensure_context_problem_type(self, expected: str) -> None:
        actual = str(self.context.get("problem_type", "") or "").lower()
        if actual != str(expected).lower():
            raise ValueError(f"Provider context problem_type mismatch: expected {expected}, got {actual}")

    def _create_model(self, seed: int):
        previous_dtype = self.paddle.get_default_dtype()
        self.paddle.set_default_dtype("float64")
        try:
            model = _PaddleSymbolicPDENet(
                self.paddle,
                int(self.context["n_dim"]),
                self.context["hidden_dims"],
                int(self.context["n_eqs"]),
                seed=seed,
            )
        finally:
            self.paddle.set_default_dtype(previous_dtype)
        model.eval()
        return model

    def _train_stationary(self) -> Dict[str, Any]:
        best_loss = float("inf")
        best_state = self.model.state_dict()
        best_retry = -1

        for retry in range(max(int(self.context["max_retries"]), 0) + 1):
            self.model = self._create_model(int(self.context["seed"]) + retry)
            optimizer = self.paddle.optimizer.Adam(
                learning_rate=float(self.context["learning_rate"]),
                parameters=self.model.parameters(),
            )
            run_best_loss = float("inf")
            run_best_state = self.model.state_dict()
            for _epoch in range(max(int(self.context["epochs_adam"]), 0)):
                optimizer.clear_grad()
                loss = self._loss_fn_stationary()
                loss.backward()
                optimizer.step()
                loss_scalar = float(loss.detach().cpu().numpy())
                if loss_scalar < run_best_loss:
                    run_best_loss = loss_scalar
                    run_best_state = self.model.state_dict()
                if loss_scalar < float(self.context["dnn_tol"]):
                    break
            if run_best_loss < best_loss:
                best_loss = run_best_loss
                best_state = run_best_state
                best_retry = retry
            if run_best_loss < float(self.context["dnn_tol"]):
                break

        self.model.set_state_dict(best_state)
        self.model.eval()
        return {"final_loss": float(best_loss), "retry": int(best_retry)}

    def _train_time_step(self, *, dt: float, U_current: np.ndarray) -> Dict[str, Any]:
        best_loss = float("inf")
        best_state = self.model.state_dict()
        best_retry = -1
        tolerance = float(self.context["dnn_tol"]) * float(dt)
        U_current = np.asarray(U_current, dtype=np.float64)

        for retry in range(max(int(self.context["max_retries"]), 0) + 1):
            self.model = self._create_model(int(self.context["seed"]) + retry)
            optimizer = self.paddle.optimizer.Adam(
                learning_rate=float(self.context["learning_rate"]),
                parameters=self.model.parameters(),
            )
            run_best_loss = float("inf")
            run_best_state = self.model.state_dict()
            for _epoch in range(max(int(self.context["epochs_adam"]), 0)):
                optimizer.clear_grad()
                loss = self._loss_fn_time(U_current=U_current, dt=float(dt))
                loss.backward()
                optimizer.step()
                loss_scalar = float(loss.detach().cpu().numpy())
                if loss_scalar < run_best_loss:
                    run_best_loss = loss_scalar
                    run_best_state = self.model.state_dict()
                if loss_scalar < tolerance:
                    break
            if run_best_loss < best_loss:
                best_loss = run_best_loss
                best_state = run_best_state
                best_retry = retry
            if run_best_loss < tolerance:
                break

        self.model.set_state_dict(best_state)
        self.model.eval()
        return {"final_loss": float(best_loss), "retry": int(best_retry)}

    def _hidden_batch_fn(self, points):
        self.model.eval()
        hidden, _ = self.model(points)
        return hidden

    def _output_batch_fn(self, points):
        self.model.eval()
        _, output = self.model(points)
        return output

    def _build_term_evaluators(self) -> Dict[str, list[Callable[[Dict[str, Any]], Any]]]:
        return {
            key: [self._compile_expression(term.get("symbolic_expr", "0")) for term in self.context["operator_terms"].get(key, [])]
            for key in ("L1", "L2", "F", "N", "S")
        }

    def _compile_expression(self, expression: str) -> Callable[[Dict[str, Any]], Any]:
        expr = str(expression or "0").strip() or "0"
        tokens = set(_TOKEN_PATTERN.findall(expr))
        constants = {
            name: float(value)
            for name, value in (self.context.get("constants", {}) or {}).items()
        }

        def evaluator(local_vars: Dict[str, Any]):
            env: Dict[str, Any] = {
                "sin": self.paddle.sin,
                "cos": self.paddle.cos,
                "tan": self.paddle.tan,
                "exp": self.paddle.exp,
                "log": self.paddle.log,
                "sqrt": self.paddle.sqrt,
                "sinh": self.paddle.sinh,
                "cosh": self.paddle.cosh,
                "tanh": self.paddle.tanh,
                "abs": self.paddle.abs,
                "pi": self.paddle.to_tensor(constants.get("pi", np.pi), dtype="float64"),
            }
            for name, value in constants.items():
                env[name] = self.paddle.to_tensor(value, dtype="float64")
            for name in tokens:
                if name in env:
                    continue
                if name in local_vars:
                    env[name] = local_vars[name]
            return eval(expr, {"__builtins__": {}}, env)

        return evaluator

    def _points_to_tensor(self, points: np.ndarray, *, requires_grad: bool):
        tensor = self.paddle.to_tensor(np.asarray(points, dtype=np.float64), dtype="float64")
        tensor.stop_gradient = not requires_grad
        return tensor

    def _evaluate_symbol(self, x_tensor, symbol_name: str):
        info = self.context["symbol_info"][symbol_name]
        var_idx = int(info["var_idx"])
        pattern = tuple(int(order) for order in info["derivative_pattern"])
        _, output = self.model(x_tensor)
        current = output[:, var_idx : var_idx + 1]
        if all(order == 0 for order in pattern):
            return current.reshape([-1])
        for axis, order in enumerate(pattern):
            for _ in range(order):
                current = self.paddle.grad(
                    outputs=current,
                    inputs=x_tensor,
                    grad_outputs=self.paddle.ones_like(current),
                    create_graph=True,
                    retain_graph=True,
                )[0][:, axis : axis + 1]
        return current.reshape([-1])

    def _evaluate_operator(self, x_tensor, op_name: str):
        batch_size = int(x_tensor.shape[0])
        local_vars: Dict[str, Any] = {
            name: x_tensor[:, dim_idx]
            for dim_idx, name in enumerate(self.context["spatial_vars"])
        }
        for symbol_name in self.context["symbol_info"]:
            local_vars[symbol_name] = self._evaluate_symbol(x_tensor, symbol_name)

        outputs = []
        for evaluator in self._term_evaluators.get(op_name, []):
            value = evaluator(local_vars)
            if np.isscalar(value):
                value = self.paddle.full([batch_size], float(value), dtype="float64")
            outputs.append(self.paddle.reshape(value, [-1]))
        return outputs

    def _pad_operator_outputs(self, outputs: Dict[str, list[Any]], batch_size: int) -> Dict[str, list[Any]]:
        template = self.paddle.zeros([batch_size], dtype="float64")
        n_eqs = int(self.context["n_eqs"])
        padded: Dict[str, list[Any]] = {}
        for key in ("L1", "L2", "F", "N", "S"):
            values = list(outputs.get(key, []))
            if len(values) < n_eqs:
                values.extend([template] * (n_eqs - len(values)))
            padded[key] = values[:n_eqs]
        return padded

    def _gradient_matrix(self, x_tensor, var_idx: int):
        _, output = self.model(x_tensor)
        current = output[:, int(var_idx) : int(var_idx) + 1]
        grads = self.paddle.grad(
            outputs=current,
            inputs=x_tensor,
            grad_outputs=self.paddle.ones_like(current),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grads

    def _compute_boundary_loss(self):
        total_error = self.paddle.to_tensor(0.0, dtype="float64")
        total_weight = self.paddle.to_tensor(0.0, dtype="float64")
        for var_idx, entry in (self._boundary_program or {}).items():
            dirichlet = entry.get("dirichlet", {})
            x_dir = np.asarray(dirichlet.get("x", []), dtype=np.float64)
            if x_dir.size > 0:
                x_tensor = self._points_to_tensor(x_dir, requires_grad=False)
                _, pred = self.model(x_tensor)
                pred = pred[:, int(var_idx) : int(var_idx) + 1]
                values = self.paddle.to_tensor(np.asarray(dirichlet.get("values", []), dtype=np.float64), dtype="float64")
                weights_np = np.asarray(dirichlet.get("weights", []), dtype=np.float64)
                weights = self.paddle.to_tensor(weights_np, dtype="float64") if weights_np.size > 0 else self.paddle.ones_like(values)
                total_error = total_error + self.paddle.sum(weights * (pred - values) ** 2)
                total_weight = total_weight + self.paddle.sum(weights)

            neumann = entry.get("neumann", {})
            x_neu = np.asarray(neumann.get("x", []), dtype=np.float64)
            if x_neu.size > 0:
                x_tensor = self._points_to_tensor(x_neu, requires_grad=True)
                grads = self._gradient_matrix(x_tensor, int(var_idx))
                normals = self.paddle.to_tensor(np.asarray(neumann.get("normals", []), dtype=np.float64), dtype="float64")
                values = self.paddle.to_tensor(np.asarray(neumann.get("values", []), dtype=np.float64), dtype="float64")
                normal_derivative = self.paddle.sum(grads * normals, axis=1, keepdim=True)
                weights_np = np.asarray(neumann.get("weights", []), dtype=np.float64)
                weights = self.paddle.to_tensor(weights_np, dtype="float64") if weights_np.size > 0 else self.paddle.ones_like(values)
                total_error = total_error + self.paddle.sum(weights * (normal_derivative - values) ** 2)
                total_weight = total_weight + self.paddle.sum(weights)

            robin = entry.get("robin", {})
            x_rob = np.asarray(robin.get("x", []), dtype=np.float64)
            if x_rob.size > 0:
                x_tensor = self._points_to_tensor(x_rob, requires_grad=True)
                _, output = self.model(x_tensor)
                current = output[:, int(var_idx) : int(var_idx) + 1]
                grads = self._gradient_matrix(x_tensor, int(var_idx))
                normals = self.paddle.to_tensor(np.asarray(robin.get("normals", []), dtype=np.float64), dtype="float64")
                values = self.paddle.to_tensor(np.asarray(robin.get("values", []), dtype=np.float64), dtype="float64")
                params_tuple = robin.get("params") or (1.0, 1.0)
                alpha = float(params_tuple[0])
                beta = float(params_tuple[1]) if len(params_tuple) > 1 else 1.0
                normal_derivative = self.paddle.sum(grads * normals, axis=1, keepdim=True)
                robin_value = alpha * current + beta * normal_derivative
                weights_np = np.asarray(robin.get("weights", []), dtype=np.float64)
                weights = self.paddle.to_tensor(weights_np, dtype="float64") if weights_np.size > 0 else self.paddle.ones_like(values)
                total_error = total_error + self.paddle.sum(weights * (robin_value - values) ** 2)
                total_weight = total_weight + self.paddle.sum(weights)

            for pair in entry.get("periodic", {}).get("pairs", []) or []:
                x_1 = np.asarray(pair.get("x_1", []), dtype=np.float64)
                x_2 = np.asarray(pair.get("x_2", []), dtype=np.float64)
                if x_1.size == 0 or x_2.size == 0:
                    continue
                x_1_tensor = self._points_to_tensor(x_1, requires_grad=False)
                x_2_tensor = self._points_to_tensor(x_2, requires_grad=False)
                _, pred_1 = self.model(x_1_tensor)
                _, pred_2 = self.model(x_2_tensor)
                pred_1 = pred_1[:, int(var_idx) : int(var_idx) + 1]
                pred_2 = pred_2[:, int(var_idx) : int(var_idx) + 1]
                pair_weight = float(pair.get("weight", 1.0))
                total_error = total_error + pair_weight * self.paddle.sum((pred_1 - pred_2) ** 2)
                total_weight = total_weight + pair_weight * float(pred_1.numel())

        if float(total_weight.detach().cpu().numpy()) <= 0.0:
            return self.paddle.to_tensor(0.0, dtype="float64")
        return total_error / total_weight

    def _loss_fn_stationary(self):
        x_tensor = self._points_to_tensor(self._x_train, requires_grad=True)
        outputs = {
            key: self._evaluate_operator(x_tensor, key)
            for key in ("L1", "L2", "F", "N", "S")
        }
        outputs = self._pad_operator_outputs(outputs, int(x_tensor.shape[0]))
        pde_loss = self.paddle.to_tensor(0.0, dtype="float64")
        for eq_idx in range(int(self.context["n_eqs"])):
            residual = outputs["L1"][eq_idx] - outputs["S"][eq_idx]
            pde_loss = pde_loss + self.paddle.mean(residual**2)
        boundary_loss = self._compute_boundary_loss()
        return pde_loss + float(self.context["boundary_constraint_weight"]) * boundary_loss

    def _loss_fn_time(self, *, U_current: np.ndarray, dt: float):
        x_tensor = self._points_to_tensor(self._x_train, requires_grad=True)
        outputs = {
            key: self._evaluate_operator(x_tensor, key)
            for key in ("L1", "L2", "F", "N", "S")
        }
        outputs = self._pad_operator_outputs(outputs, int(x_tensor.shape[0]))
        _, predicted = self.model(x_tensor)
        U_current_tensor = self.paddle.to_tensor(np.asarray(U_current, dtype=np.float64), dtype="float64")
        pde_loss = self.paddle.to_tensor(0.0, dtype="float64")
        for eq_idx in range(int(self.context["n_eqs"])):
            spatial_residual = (
                outputs["L1"][eq_idx]
                + outputs["L2"][eq_idx] * outputs["F"][eq_idx]
                + outputs["N"][eq_idx]
                - outputs["S"][eq_idx]
            )
            total_residual = (predicted[:, eq_idx] - U_current_tensor[:, eq_idx]) + float(dt) * spatial_residual
            pde_loss = pde_loss + self.paddle.mean(total_residual**2)
        boundary_loss = self._compute_boundary_loss()
        return pde_loss + float(self.context["boundary_constraint_weight"]) * boundary_loss


TrainablePaddleLinearPDEProvider = TrainablePaddleSymbolicPDEProvider
