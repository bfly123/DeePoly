from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.abstract_class.spotter_feature_provider import JaxSpotterFeatureProvider
from src.spotter_runtime.backends.func_fitting_common import build_func_fitting_context


class TrainableJaxFuncFittingProvider(JaxSpotterFeatureProvider):
    backend_name = "jax"

    def __init__(self, *, config=None, data_train: Optional[Dict[str, Any]] = None, program: Optional[Dict[str, Any]] = None):
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import optax

        self.context = dict(program or build_func_fitting_context(config, data_train))
        self.jax = jax
        self.jnp = jnp
        self.optax = optax
        self.params = self._init_params(self.context["seed"])
        self.feature_dim = int(self.context["hidden_dims"][-1])

        super().__init__(
            hidden_fn=lambda point: self._forward_hidden(self.params, point),
            output_fn=lambda point: self._forward_output(self.params, point),
            feature_dim=self.feature_dim,
        )
        self._loss_and_grad = self.jax.jit(self.jax.value_and_grad(self._loss_fn))

    def supports_external_training(self) -> bool:
        return True

    def train_external(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if str(request.get("problem_type", "")).lower() != "func_fitting":
            raise ValueError("JAX func_fitting provider only supports problem_type=func_fitting")
        x_train = self.jnp.asarray(np.asarray(request.get("x_train", self.context["x_train"]), dtype=np.float64), dtype=self.jnp.float64)
        y_train = self.jnp.asarray(np.asarray(request.get("targets", self.context["y_train"]), dtype=np.float64), dtype=self.jnp.float64)
        best_loss = float("inf")
        best_params = self.params
        best_retry = -1

        for retry in range(max(int(self.context["max_retries"]), 0) + 1):
            params = self._init_params(self.context["seed"] + retry)
            optimizer = self.optax.adam(float(self.context["learning_rate"]))
            opt_state = optimizer.init(params)
            run_best_loss = float("inf")
            run_best_params = params
            for _epoch in range(max(int(self.context["epochs_adam"]), 0)):
                loss_value, grads = self._loss_and_grad(params, x_train, y_train)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = self.optax.apply_updates(params, updates)
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

    def _init_params(self, seed: int):
        key = self.jax.random.PRNGKey(int(seed))
        dims = [int(self.context["n_dim"])] + list(self.context["hidden_dims"])
        params = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            key, w_key, b_key = self.jax.random.split(key, 3)
            limit = 1.0 / np.sqrt(float(fan_in))
            params.append({
                "w": self.jax.random.uniform(w_key, shape=(fan_in, fan_out), minval=-limit, maxval=limit, dtype=self.jnp.float64),
                "b": self.jax.random.uniform(b_key, shape=(fan_out,), minval=-limit, maxval=limit, dtype=self.jnp.float64),
            })
        key, out_key = self.jax.random.split(key)
        limit = 1.0 / np.sqrt(float(dims[-1]))
        out = {
            "w": self.jax.random.uniform(out_key, shape=(dims[-1], int(self.context["n_eqs"])), minval=-limit, maxval=limit, dtype=self.jnp.float64),
        }
        return {"layers": tuple(params), "out": out}

    def _forward_hidden(self, params, point):
        z = self.jnp.asarray(point, dtype=self.jnp.float64).reshape(1, -1)
        for layer in params["layers"]:
            z = self.jnp.tanh(z @ layer["w"] + layer["b"])
        return z.reshape(-1)

    def _forward_output(self, params, point):
        hidden = self._forward_hidden(params, point)
        return (hidden @ params["out"]["w"]).reshape(-1)

    def _loss_fn(self, params, x_train, y_train):
        pred = self.jax.vmap(lambda point: self._forward_output(params, point))(x_train)
        return self.jnp.mean((pred - y_train) ** 2)
