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
        self._start_worker()

    def _start_worker(self) -> None:
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            self._repo_root if not existing_pythonpath else f"{self._repo_root}:{existing_pythonpath}"
        )
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

    def _request(self, op: str, **payload) -> np.ndarray:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Subprocess spotter provider is not running")
        send_message(self._process.stdin, {"op": op, "payload": payload})
        response = recv_message(self._process.stdout)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", f"Subprocess spotter call failed: {op}"))
        return np.asarray(response["result"], dtype=np.float64)

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
