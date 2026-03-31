#!/usr/bin/env python3
"""Export a JAX-PI Allen-Cahn checkpoint to a DeePoly transfer field."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore a JAX-PI Allen-Cahn checkpoint and export predictions on the "
            "current DeePoly train/test grids."
        )
    )
    parser.add_argument(
        "--jaxpi-root",
        type=Path,
        default=ROOT / "tmp" / "jaxpi_pirate",
        help="Path to the local JAX-PI pirate branch checkout.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Target DeePoly case directory. Exported coordinates follow this case grid.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="JAX-PI workdir that contains <wandb.name>/ckpt.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JAX-PI Allen-Cahn config file. Defaults to examples/allen_cahn/configs/pirate.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output npz path. Defaults to <case-dir>/results/jaxpi_transfer.npz.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also export predictions on DeePoly test points as x_test/U_test.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used to build the DeePoly case grid. Default: cpu.",
    )
    return parser.parse_args()


def load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def pushd(path: Path):
    cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def build_case_grids(case_dir: Path, device: str) -> tuple[np.ndarray, np.ndarray]:
    from src.cli import create_solver, load_problem_type, resolve_case_dir

    case_dir = Path(resolve_case_dir(str(case_dir)))
    args = SimpleNamespace(
        device=device,
        linear_device=device,
        results_dir=None,
        skip_autocode=True,
        case_path=str(case_dir),
    )
    problem_type = load_problem_type(str(case_dir))
    solver = create_solver(problem_type, str(case_dir), args)
    x_train = np.asarray(solver.data_train["x"], dtype=np.float64)
    x_test = np.asarray(solver.data_test["x"], dtype=np.float64)
    return x_train, x_test


def restore_jaxpi_model(jaxpi_root: Path, config_path: Path, workdir: Path):
    sys.path.insert(0, str(jaxpi_root))
    example_dir = config_path.parent.parent

    import jax
    import jax.numpy as jnp
    from jax import vmap
    from jaxpi.utils import restore_checkpoint

    config_module = load_module("jaxpi_allen_cahn_config", config_path)
    models_module = load_module("jaxpi_allen_cahn_models", example_dir / "models.py")
    utils_module = load_module("jaxpi_allen_cahn_utils", example_dir / "utils.py")

    config = config_module.get_config()
    u_ref, t_star, x_star = utils_module.get_dataset()
    u0 = u_ref[0, :]

    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    model = models_module.AllenCahn(config, u0, t_star, x_star)
    ckpt_path = workdir / config.wandb.name / "ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"JAX-PI checkpoint directory not found: {ckpt_path}")
    model.state = restore_checkpoint(model.state, str(ckpt_path))
    params = model.state.params

    def predict(points: np.ndarray) -> np.ndarray:
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError(f"Expected points with shape (N, 2), got {points.shape}")
        t = jnp.asarray(points[:, 1])
        x = jnp.asarray(points[:, 0])
        pred = vmap(model.u_net, (None, 0, 0))(params, t, x)
        return np.asarray(jax.device_get(pred), dtype=np.float64).reshape(-1, 1)

    return predict, config


def main() -> int:
    args = parse_args()
    jaxpi_root = args.jaxpi_root.expanduser().resolve()
    case_dir = args.case_dir.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()

    if args.config is None:
        config_path = jaxpi_root / "examples" / "allen_cahn" / "configs" / "pirate.py"
    else:
        config_path = args.config.expanduser().resolve()

    if args.output is None:
        output_path = case_dir / "results" / "jaxpi_transfer.npz"
    else:
        output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, x_test = build_case_grids(case_dir, args.device)

    with pushd(config_path.parent.parent):
        predict, config = restore_jaxpi_model(jaxpi_root, config_path, workdir)
        u_train = predict(x_train)
        payload = {
            "x_train": x_train,
            "U_train": u_train,
        }
        if args.include_test:
            payload["x_test"] = x_test
            payload["U_test"] = predict(x_test)

    np.savez(
        output_path,
        **payload,
        source=str(workdir),
        config_path=str(config_path),
        wandb_name=str(config.wandb.name),
    )
    print(f"Exported JAX-PI transfer field: {output_path}")
    print(f"  train points: {x_train.shape[0]}")
    if args.include_test:
        print(f"  test points: {x_test.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
