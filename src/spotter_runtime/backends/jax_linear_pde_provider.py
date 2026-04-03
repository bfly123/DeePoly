from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.abstract_class.spotter_feature_provider import JaxSpotterFeatureProvider
from src.spotter_runtime.backends.pde_program_common import build_symbolic_pde_context


_TOKEN_PATTERN = re.compile(r"\b[A-Za-z_]\w*\b")


class TrainableJaxSymbolicPDEProvider(JaxSpotterFeatureProvider):
    backend_name = "jax"

    def __init__(self, *, config=None, data_train: Optional[Dict[str, Any]] = None, program: Optional[Dict[str, Any]] = None):
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import optax

        self.jax = jax
        self.jnp = jnp
        self.optax = optax
        self.context = dict(program or build_symbolic_pde_context(config, data_train))
        self.feature_dim = int(self.context["hidden_dims"][-1])
        self.params = self._init_params(int(self.context["seed"]))
        self._x_train = self.jnp.asarray(np.asarray(self.context["x_train"], dtype=np.float64), dtype=self.jnp.float64)
        self._boundary_program = self._convert_boundary_program(self.context["boundary_program"])
        self._term_evaluators = self._build_term_evaluators()
        self._symbol_batch_fn_cache: Dict[tuple[int, tuple[int, ...]], Callable[[Any, Any], Any]] = {}

        self._model_hidden_batch_jit = self.jax.jit(self._model_hidden_batch_impl)
        self._model_output_batch_jit = self.jax.jit(self._model_output_batch_impl)
        self._stationary_loss_and_grad = self.jax.jit(self.jax.value_and_grad(self._loss_fn_stationary))
        self._time_loss_and_grad = self.jax.jit(self.jax.value_and_grad(self._loss_fn_time))

        super().__init__(
            hidden_fn=lambda point: self._forward_hidden(self.params, point),
            output_fn=lambda point: self._forward_output(self.params, point),
            feature_dim=self.feature_dim,
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
            raise ValueError(f"JAX symbolic PDE provider does not support train_external for problem_type={problem_type}")
        self._ensure_context_problem_type("linear_pde")
        return self._train_stationary()

    def train_external_step(self, *, problem_type: str, dt: float, U_current: np.ndarray, iteration: Optional[int] = None) -> Dict[str, Any]:
        _ = iteration
        requested = str(problem_type or "").lower()
        if requested != "time_pde":
            raise ValueError(f"JAX symbolic PDE provider only supports train_external_step for time_pde. Got: {problem_type}")
        self._ensure_context_problem_type("time_pde")
        return self._train_time_step(dt=float(dt), U_current=np.asarray(U_current, dtype=np.float64))

    def model_apply(self, params, points):
        return self._model_hidden_batch_jit(params, points), self._model_output_batch_jit(params, points)

    def residual_apply(self, params, points):
        return self._residual_apply(params, points)

    def boundary_loss(self, params):
        return self._boundary_loss(params)

    def _ensure_context_problem_type(self, expected: str) -> None:
        actual = str(self.context.get("problem_type", "") or "").lower()
        if actual != str(expected).lower():
            raise ValueError(f"Provider context problem_type mismatch: expected {expected}, got {actual}")

    def _build_stationary_train_step(self, optimizer):
        @self.jax.jit
        def train_step(params, opt_state):
            loss_value, grads = self._stationary_loss_and_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = self.optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        return train_step

    def _build_time_train_step(self, optimizer):
        @self.jax.jit
        def train_step(params, opt_state, U_current, dt):
            loss_value, grads = self._time_loss_and_grad(params, U_current, dt)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = self.optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        return train_step

    def _train_stationary(self) -> Dict[str, Any]:
        best_loss = float("inf")
        best_params = self.params
        best_retry = -1

        for retry in range(max(int(self.context["max_retries"]), 0) + 1):
            params = self._init_params(int(self.context["seed"]) + retry)
            optimizer = self.optax.adam(float(self.context["learning_rate"]))
            opt_state = optimizer.init(params)
            train_step = self._build_stationary_train_step(optimizer)

            run_best_loss = float("inf")
            run_best_params = params
            for _epoch in range(max(int(self.context["epochs_adam"]), 0)):
                params, opt_state, loss_value = train_step(params, opt_state)
                loss_scalar = float(loss_value)
                if loss_scalar < run_best_loss:
                    run_best_loss = loss_scalar
                    run_best_params = params
                if loss_scalar < float(self.context["dnn_tol"]):
                    break

            if run_best_loss < best_loss:
                best_loss = run_best_loss
                best_params = run_best_params
                best_retry = retry
            if run_best_loss < float(self.context["dnn_tol"]):
                break

        self.params = best_params
        return {"final_loss": float(best_loss), "retry": int(best_retry)}

    def _train_time_step(self, *, dt: float, U_current: np.ndarray) -> Dict[str, Any]:
        best_loss = float("inf")
        best_params = self.params
        best_retry = -1
        tolerance = float(self.context["dnn_tol"]) * float(dt)
        U_current_tensor = self.jnp.asarray(np.asarray(U_current, dtype=np.float64), dtype=self.jnp.float64)

        for retry in range(max(int(self.context["max_retries"]), 0) + 1):
            params = self._init_params(int(self.context["seed"]) + retry)
            optimizer = self.optax.adam(float(self.context["learning_rate"]))
            opt_state = optimizer.init(params)
            train_step = self._build_time_train_step(optimizer)

            run_best_loss = float("inf")
            run_best_params = params
            for _epoch in range(max(int(self.context["epochs_adam"]), 0)):
                params, opt_state, loss_value = train_step(params, opt_state, U_current_tensor, float(dt))
                loss_scalar = float(loss_value)
                if loss_scalar < run_best_loss:
                    run_best_loss = loss_scalar
                    run_best_params = params
                if loss_scalar < tolerance:
                    break

            if run_best_loss < best_loss:
                best_loss = run_best_loss
                best_params = run_best_params
                best_retry = retry
            if run_best_loss < tolerance:
                break

        self.params = best_params
        return {"final_loss": float(best_loss), "retry": int(best_retry)}

    def _init_params(self, seed: int):
        key = self.jax.random.PRNGKey(int(seed))
        dims = [int(self.context["n_dim"])] + list(self.context["hidden_dims"])
        params = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            key, w_key, b_key = self.jax.random.split(key, 3)
            limit = 1.0 / np.sqrt(float(fan_in))
            params.append(
                {
                    "w": self.jax.random.uniform(
                        w_key,
                        shape=(fan_in, fan_out),
                        minval=-limit,
                        maxval=limit,
                        dtype=self.jnp.float64,
                    ),
                    "b": self.jax.random.uniform(
                        b_key,
                        shape=(fan_out,),
                        minval=-limit,
                        maxval=limit,
                        dtype=self.jnp.float64,
                    ),
                }
            )
        key, out_key = self.jax.random.split(key)
        limit = 1.0 / np.sqrt(float(dims[-1]))
        out = {
            "w": self.jax.random.uniform(
                out_key,
                shape=(dims[-1], int(self.context["n_eqs"])),
                minval=-limit,
                maxval=limit,
                dtype=self.jnp.float64,
            )
        }
        return {"layers": tuple(params), "out": out}

    def _forward_hidden(self, params, point):
        z = self.jnp.asarray(point, dtype=self.jnp.float64)
        for layer in params["layers"]:
            z = self.jnp.tanh(z @ layer["w"] + layer["b"])
        return z

    def _forward_output(self, params, point):
        hidden = self._forward_hidden(params, point)
        return hidden @ params["out"]["w"]

    def _model_hidden_batch_impl(self, params, points):
        return self.jax.vmap(lambda point: self._forward_hidden(params, point))(points)

    def _model_output_batch_impl(self, params, points):
        return self.jax.vmap(lambda point: self._forward_output(params, point))(points)

    def _convert_boundary_program(self, boundary_program: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        converted: Dict[int, Dict[str, Any]] = {}
        for var_idx, entry in (boundary_program or {}).items():
            converted[int(var_idx)] = {
                "dirichlet": self._convert_standard_boundary(entry.get("dirichlet", {})),
                "neumann": self._convert_standard_boundary(entry.get("neumann", {}), include_normals=True),
                "robin": self._convert_standard_boundary(entry.get("robin", {}), include_normals=True, include_params=True),
                "periodic": {
                    "pairs": [
                        {
                            "x_1": self.jnp.asarray(np.asarray(pair.get("x_1", []), dtype=np.float64), dtype=self.jnp.float64),
                            "x_2": self.jnp.asarray(np.asarray(pair.get("x_2", []), dtype=np.float64), dtype=self.jnp.float64),
                            "weight": float(pair.get("weight", 1.0)),
                        }
                        for pair in entry.get("periodic", {}).get("pairs", []) or []
                    ]
                },
            }
        return converted

    def _convert_standard_boundary(
        self,
        bc_data: Dict[str, Any],
        *,
        include_normals: bool = False,
        include_params: bool = False,
    ) -> Dict[str, Any]:
        converted = {
            "x": self.jnp.asarray(np.asarray(bc_data.get("x", []), dtype=np.float64), dtype=self.jnp.float64),
            "values": self.jnp.asarray(np.asarray(bc_data.get("values", []), dtype=np.float64), dtype=self.jnp.float64),
            "weights": self.jnp.asarray(np.asarray(bc_data.get("weights", []), dtype=np.float64), dtype=self.jnp.float64),
        }
        if include_normals:
            converted["normals"] = self.jnp.asarray(np.asarray(bc_data.get("normals", []), dtype=np.float64), dtype=self.jnp.float64)
        if include_params:
            converted["params"] = bc_data.get("params")
        return converted

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
                "sin": self.jnp.sin,
                "cos": self.jnp.cos,
                "tan": self.jnp.tan,
                "exp": self.jnp.exp,
                "log": self.jnp.log,
                "sqrt": self.jnp.sqrt,
                "sinh": self.jnp.sinh,
                "cosh": self.jnp.cosh,
                "tanh": self.jnp.tanh,
                "abs": self.jnp.abs,
                "pi": self.jnp.asarray(constants.get("pi", np.pi), dtype=self.jnp.float64),
            }
            for name, value in constants.items():
                env[name] = self.jnp.asarray(value, dtype=self.jnp.float64)
            for name in tokens:
                if name in env:
                    continue
                if name in local_vars:
                    env[name] = local_vars[name]
            return eval(expr, {"__builtins__": {}}, env)

        return evaluator

    def _build_scalar_derivative_fn(self, *, var_idx: int, pattern: tuple[int, ...]):
        def scalar_fn(params, point):
            return self._forward_output(params, point)[int(var_idx)]

        current_fn = scalar_fn
        for axis, order in enumerate(pattern):
            for _ in range(int(order)):
                previous_fn = current_fn
                current_fn = lambda params, point, previous_fn=previous_fn, axis=axis: self.jax.grad(
                    lambda local_point: previous_fn(params, local_point)
                )(point)[axis]
        return current_fn

    def _get_symbol_batch_fn(self, *, var_idx: int, pattern: tuple[int, ...]):
        key = (int(var_idx), tuple(int(order) for order in pattern))
        cached = self._symbol_batch_fn_cache.get(key)
        if cached is not None:
            return cached
        scalar_fn = self._build_scalar_derivative_fn(var_idx=int(var_idx), pattern=tuple(pattern))
        compiled = self.jax.jit(lambda params, points, scalar_fn=scalar_fn: self.jax.vmap(lambda point: scalar_fn(params, point))(points))
        self._symbol_batch_fn_cache[key] = compiled
        return compiled

    def _evaluate_symbol_values(self, params, points) -> Dict[str, Any]:
        local_vars: Dict[str, Any] = {
            name: points[:, dim_idx]
            for dim_idx, name in enumerate(self.context["spatial_vars"])
        }
        output_cache = None
        for symbol_name, info in self.context["symbol_info"].items():
            pattern = tuple(int(order) for order in info["derivative_pattern"])
            if all(order == 0 for order in pattern):
                if output_cache is None:
                    output_cache = self._model_output_batch_jit(params, points)
                local_vars[symbol_name] = output_cache[:, int(info["var_idx"])]
                continue
            batch_fn = self._get_symbol_batch_fn(var_idx=int(info["var_idx"]), pattern=pattern)
            local_vars[symbol_name] = batch_fn(params, points)
        return local_vars

    def _evaluate_operator_terms(self, local_vars: Dict[str, Any], points, op_name: str):
        batch_size = int(points.shape[0])
        outputs = []
        for evaluator in self._term_evaluators.get(op_name, []):
            value = evaluator(local_vars)
            if np.isscalar(value):
                value = self.jnp.full((batch_size,), float(value), dtype=self.jnp.float64)
            outputs.append(self.jnp.asarray(value, dtype=self.jnp.float64).reshape(-1))
        return outputs

    def _pad_operator_outputs(self, outputs: Dict[str, list[Any]], batch_size: int) -> Dict[str, list[Any]]:
        padded: Dict[str, list[Any]] = {}
        template = self.jnp.zeros((batch_size,), dtype=self.jnp.float64)
        n_eqs = int(self.context["n_eqs"])
        for key in ("L1", "L2", "F", "N", "S"):
            values = list(outputs.get(key, []))
            if len(values) < n_eqs:
                values.extend([template] * (n_eqs - len(values)))
            padded[key] = values[:n_eqs]
        return padded

    def _residual_apply(self, params, points):
        local_vars = self._evaluate_symbol_values(params, points)
        operator_outputs = {
            key: self._evaluate_operator_terms(local_vars, points, key)
            for key in ("L1", "L2", "F", "N", "S")
        }
        operator_outputs = self._pad_operator_outputs(operator_outputs, int(points.shape[0]))
        residuals = []
        mode = str(self.context.get("residual_mode", "linear_pde")).lower()
        for eq_idx in range(int(self.context["n_eqs"])):
            if mode == "linear_pde":
                residual = operator_outputs["L1"][eq_idx] - operator_outputs["S"][eq_idx]
            else:
                residual = (
                    operator_outputs["L1"][eq_idx]
                    + operator_outputs["L2"][eq_idx] * operator_outputs["F"][eq_idx]
                    + operator_outputs["N"][eq_idx]
                    - operator_outputs["S"][eq_idx]
                )
            residuals.append(residual)
        return self.jnp.stack(residuals, axis=1)

    def _gradient_matrix(self, params, points, var_idx: int):
        components = []
        for axis in range(int(self.context["n_dim"])):
            pattern = tuple(1 if idx == axis else 0 for idx in range(int(self.context["n_dim"])))
            batch_fn = self._get_symbol_batch_fn(var_idx=int(var_idx), pattern=pattern)
            components.append(batch_fn(params, points))
        return self.jnp.stack(components, axis=1)

    def _boundary_loss(self, params):
        total_error = self.jnp.asarray(0.0, dtype=self.jnp.float64)
        total_weight = self.jnp.asarray(0.0, dtype=self.jnp.float64)

        for var_idx, entry in self._boundary_program.items():
            dirichlet = entry.get("dirichlet", {})
            if dirichlet["x"].size > 0:
                pred = self._model_output_batch_jit(params, dirichlet["x"])[:, int(var_idx) : int(var_idx) + 1]
                weights = dirichlet["weights"] if dirichlet["weights"].size > 0 else self.jnp.ones_like(dirichlet["values"])
                total_error = total_error + self.jnp.sum(weights * (pred - dirichlet["values"]) ** 2)
                total_weight = total_weight + self.jnp.sum(weights)

            neumann = entry.get("neumann", {})
            if neumann["x"].size > 0:
                grad_matrix = self._gradient_matrix(params, neumann["x"], int(var_idx))
                normal_derivative = self.jnp.sum(grad_matrix * neumann["normals"], axis=1, keepdims=True)
                weights = neumann["weights"] if neumann["weights"].size > 0 else self.jnp.ones_like(neumann["values"])
                total_error = total_error + self.jnp.sum(weights * (normal_derivative - neumann["values"]) ** 2)
                total_weight = total_weight + self.jnp.sum(weights)

            robin = entry.get("robin", {})
            if robin["x"].size > 0:
                pred = self._model_output_batch_jit(params, robin["x"])[:, int(var_idx) : int(var_idx) + 1]
                grad_matrix = self._gradient_matrix(params, robin["x"], int(var_idx))
                normal_derivative = self.jnp.sum(grad_matrix * robin["normals"], axis=1, keepdims=True)
                params_tuple = robin.get("params") or (1.0, 1.0)
                alpha = float(params_tuple[0])
                beta = float(params_tuple[1]) if len(params_tuple) > 1 else 1.0
                robin_value = alpha * pred + beta * normal_derivative
                weights = robin["weights"] if robin["weights"].size > 0 else self.jnp.ones_like(robin["values"])
                total_error = total_error + self.jnp.sum(weights * (robin_value - robin["values"]) ** 2)
                total_weight = total_weight + self.jnp.sum(weights)

            for pair in entry.get("periodic", {}).get("pairs", []) or []:
                if pair["x_1"].size == 0 or pair["x_2"].size == 0:
                    continue
                pred_1 = self._model_output_batch_jit(params, pair["x_1"])[:, int(var_idx) : int(var_idx) + 1]
                pred_2 = self._model_output_batch_jit(params, pair["x_2"])[:, int(var_idx) : int(var_idx) + 1]
                pair_weight = float(pair.get("weight", 1.0))
                total_error = total_error + pair_weight * self.jnp.sum((pred_1 - pred_2) ** 2)
                total_weight = total_weight + pair_weight * float(pred_1.size)

        return self.jnp.where(total_weight > 0, total_error / total_weight, self.jnp.asarray(0.0, dtype=self.jnp.float64))

    def _loss_fn_stationary(self, params):
        residuals = self._residual_apply(params, self._x_train)
        pde_loss = self.jnp.mean(residuals**2)
        bc_loss = self._boundary_loss(params)
        return pde_loss + float(self.context["boundary_constraint_weight"]) * bc_loss

    def _loss_fn_time(self, params, U_current, dt):
        spatial_residual = self._residual_apply(params, self._x_train)
        outputs = self._model_output_batch_jit(params, self._x_train)
        dt_value = self.jnp.asarray(dt, dtype=self.jnp.float64)
        total_residual = (outputs - U_current) + dt_value * spatial_residual
        pde_loss = self.jnp.mean(total_residual**2)
        bc_loss = self._boundary_loss(params)
        return pde_loss + float(self.context["boundary_constraint_weight"]) * bc_loss


TrainableJaxLinearPDEProvider = TrainableJaxSymbolicPDEProvider
