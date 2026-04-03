from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.abstract_class.spotter_feature_provider import create_spotter_feature_provider_from_settings
from src.spotter_runtime.ipc import recv_message, send_message


def _redirect_prints_to_stderr():
    protocol_stdout = sys.stdout.buffer
    sys.stdout = sys.stderr
    return protocol_stdout


def _build_runtime_namespace(config_payload):
    return SimpleNamespace(**dict(config_payload or {}))


def main() -> int:
    protocol_stdout = _redirect_prints_to_stderr()
    provider = None
    stdin = sys.stdin.buffer
    stdout = protocol_stdout

    while True:
        try:
            message = recv_message(stdin)
        except EOFError:
            return 0
        op = message.get("op")
        payload = message.get("payload", {})

        try:
            if op == "startup":
                repo_root = payload.get("repo_root")
                if repo_root:
                    repo_path = str(Path(repo_root).resolve())
                    if repo_path not in sys.path:
                        sys.path.insert(0, repo_path)
                config_ns = _build_runtime_namespace(payload.get("config"))
                provider = create_spotter_feature_provider_from_settings(
                    backend=payload["candidate"]["backend"],
                    provider_cfg=payload["candidate"].get("provider_cfg", {}),
                    case_dir=payload.get("case_dir"),
                    config=config_ns,
                    model=None,
                    data_train=payload.get("data_train"),
                    data_test=payload.get("data_test"),
                    device=payload["candidate"].get("device") or getattr(config_ns, "device", "cpu"),
                )
                supports = getattr(provider, "supports_external_training", None)
                supports_external_training = bool(supports() if callable(supports) else supports)
                send_message(
                    stdout,
                    {
                        "ok": True,
                        "backend_name": getattr(provider, "backend_name", "custom"),
                        "supports_external_training": supports_external_training,
                    },
                )
                continue

            if op == "shutdown":
                send_message(stdout, {"ok": True})
                return 0

            if provider is None:
                raise RuntimeError("Worker has not been started with a provider")

            if op == "hidden_features":
                result = provider.hidden_features(
                    np.asarray(payload["x_norm"], dtype=np.float64),
                    np.asarray(payload["x_min"], dtype=np.float64),
                    np.asarray(payload["x_max"], dtype=np.float64),
                )
            elif op == "hidden_feature_derivatives":
                result = provider.hidden_feature_derivatives(
                    np.asarray(payload["x_norm"], dtype=np.float64),
                    np.asarray(payload["x_min"], dtype=np.float64),
                    np.asarray(payload["x_max"], dtype=np.float64),
                    np.asarray(payload["derivative"], dtype=int),
                )
            elif op == "predict_output":
                result = provider.predict_output(np.asarray(payload["points"], dtype=np.float64))
            elif op == "train_external_step":
                result = provider.train_external_step(
                    problem_type=str(payload["problem_type"]),
                    dt=float(payload["dt"]),
                    U_current=np.asarray(payload["U_current"], dtype=np.float64),
                    iteration=payload.get("iteration"),
                )
            elif op == "train_external":
                result = provider.train_external(dict(payload.get("request", {}) or {}))
            else:
                raise ValueError(f"Unsupported worker operation: {op}")

            if isinstance(result, np.ndarray):
                result = np.asarray(result, dtype=np.float64)
            send_message(stdout, {"ok": True, "result": result})
        except Exception as exc:
            send_message(stdout, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
