from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.abstract_class.spotter_feature_provider import BaseSpotterFeatureProvider
from src.spotter_runtime.ipc import recv_message, send_message


class SubprocessSpotterFeatureProvider(BaseSpotterFeatureProvider):
    backend_name = "subprocess"

    def __init__(
        self,
        *,
        backend: str,
        python_executable: str,
        startup_payload: Dict[str, Any],
        repo_root: str,
    ):
        self.backend_name = str(backend)
        self._repo_root = str(repo_root)
        self._python_executable = str(python_executable)
        self._startup_payload = startup_payload
        self._process: Optional[subprocess.Popen] = None
        self._supports_external_training = False
        self._start_worker()

    def _start_worker(self) -> None:
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            self._repo_root if not existing_pythonpath else f"{self._repo_root}:{existing_pythonpath}"
        )
        if str(self.backend_name).lower() == "jax":
            candidate = dict(self._startup_payload.get("candidate", {}) or {})
            provider_cfg = dict(candidate.get("provider_cfg", {}) or {})
            options = dict(provider_cfg.get("options", {}) or {})
            jax_platform = str(options.get("jax_platform") or candidate.get("device") or "cpu").lower()
            if jax_platform == "gpu":
                jax_platform = "cuda"
            if jax_platform not in {"cpu", "cuda"}:
                jax_platform = "cpu"
            env.setdefault("JAX_PLATFORMS", jax_platform)
            env.setdefault("JAX_PLATFORM_NAME", jax_platform)
            env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            env.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
            cuda_visible_devices = options.get("cuda_visible_devices")
            if cuda_visible_devices is not None:
                env.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_visible_devices))
            elif jax_platform == "cpu":
                env.setdefault("CUDA_VISIBLE_DEVICES", "")
        command = [
            self._python_executable,
            "-m",
            "src.spotter_runtime.workers.worker_main",
        ]
        self._process = subprocess.Popen(
            command,
            cwd=self._repo_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        send_message(self._process.stdin, {"op": "startup", "payload": self._startup_payload})
        response = recv_message(self._process.stdout)
        if not response.get("ok", False):
            self.close()
            raise RuntimeError(response.get("error", "Unknown subprocess spotter startup failure"))
        self._supports_external_training = bool(response.get("supports_external_training", False))

    def _request_raw(self, op: str, **payload):
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Subprocess spotter provider is not running")
        send_message(self._process.stdin, {"op": op, "payload": payload})
        response = recv_message(self._process.stdout)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", f"Subprocess spotter call failed: {op}"))
        return response["result"]

    def _request(self, op: str, **payload) -> np.ndarray:
        return np.asarray(self._request_raw(op, **payload), dtype=np.float64)

    def supports_external_training(self) -> bool:
        return bool(self._supports_external_training)

    def train_external(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._request_raw("train_external", request=request)

    def train_external_step(
        self,
        *,
        problem_type: str,
        dt: float,
        U_current: np.ndarray,
        iteration: Optional[int] = None,
    ):
        return self._request_raw(
            "train_external_step",
            problem_type=str(problem_type),
            dt=float(dt),
            U_current=np.asarray(U_current, dtype=np.float64),
            iteration=iteration,
        )

    def hidden_features(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ) -> np.ndarray:
        return self._request(
            "hidden_features",
            x_norm=np.asarray(x_norm, dtype=np.float64),
            x_min=np.asarray(x_min, dtype=np.float64),
            x_max=np.asarray(x_max, dtype=np.float64),
        )

    def hidden_feature_derivatives(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        derivative: np.ndarray,
    ) -> np.ndarray:
        return self._request(
            "hidden_feature_derivatives",
            x_norm=np.asarray(x_norm, dtype=np.float64),
            x_min=np.asarray(x_min, dtype=np.float64),
            x_max=np.asarray(x_max, dtype=np.float64),
            derivative=np.asarray(derivative, dtype=int),
        )

    def predict_output(self, points: np.ndarray) -> np.ndarray:
        return self._request("predict_output", points=np.asarray(points, dtype=np.float64))

    def close(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.stdin is not None and self._process.stdout is not None:
                send_message(self._process.stdin, {"op": "shutdown", "payload": {}})
                recv_message(self._process.stdout)
        except Exception:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=2.0)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass
        self._process = None

    def __del__(self) -> None:  # pragma: no cover - cleanup safety
        try:
            self.close()
        except Exception:
            pass
