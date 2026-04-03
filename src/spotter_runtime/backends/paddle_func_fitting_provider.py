from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.abstract_class.spotter_feature_provider import PaddleSpotterFeatureProvider
from src.spotter_runtime.backends.func_fitting_common import build_func_fitting_context


class _PaddleFuncFittingNet:
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


class TrainablePaddleFuncFittingProvider(PaddleSpotterFeatureProvider):
    backend_name = "paddle"

    def __init__(self, *, config=None, data_train: Optional[Dict[str, Any]] = None, program: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        import paddle

        self.context = dict(program or build_func_fitting_context(config, data_train))
        self.paddle = paddle
        if device is not None:
            paddle.set_device(str(device))
        self.model = self._create_model(self.context["seed"])
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
        if str(request.get("problem_type", "")).lower() != "func_fitting":
            raise ValueError("Paddle func_fitting provider only supports problem_type=func_fitting")
        x_train = np.asarray(request.get("x_train", self.context["x_train"]), dtype=np.float64)
        y_train = np.asarray(request.get("targets", self.context["y_train"]), dtype=np.float64)
        best_loss = float("inf")
        best_state = self.model.state_dict()
        best_retry = -1

        for retry in range(max(int(self.context["max_retries"]), 0) + 1):
            self.model = self._create_model(self.context["seed"] + retry)
            optimizer = self.paddle.optimizer.Adam(
                learning_rate=float(self.context["learning_rate"]),
                parameters=self.model.parameters(),
            )
            run_best_loss = float("inf")
            run_best_state = self.model.state_dict()
            x_tensor = self.paddle.to_tensor(x_train, dtype="float64")
            y_tensor = self.paddle.to_tensor(y_train, dtype="float64")
            x_tensor.stop_gradient = True
            y_tensor.stop_gradient = True
            for _epoch in range(max(int(self.context["epochs_adam"]), 0)):
                optimizer.clear_grad()
                self.model.train()
                _, pred = self.model(x_tensor)
                loss = self.paddle.mean((pred - y_tensor) ** 2)
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

    def _create_model(self, seed: int):
        previous_dtype = self.paddle.get_default_dtype()
        self.paddle.set_default_dtype("float64")
        try:
            model = _PaddleFuncFittingNet(
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

    def _hidden_batch_fn(self, points):
        self.model.eval()
        hidden, _ = self.model(points)
        return hidden

    def _output_batch_fn(self, points):
        self.model.eval()
        _, output = self.model(points)
        return output
