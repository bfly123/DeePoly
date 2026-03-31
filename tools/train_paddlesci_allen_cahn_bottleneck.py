#!/usr/bin/env python3
"""Train a PaddleScience Allen-Cahn PirateNet with a narrow DeePoly bottleneck."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paddlesci_bottleneck_piratenet import build_bottleneck_piratenet
from tools.paddlesci_bottleneck_piratenet import load_matching_pretrained


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paddlesci-root",
        default=str(ROOT / "tmp/PaddleScience_20260329"),
        help="Path to the local PaddleScience checkout.",
    )
    parser.add_argument(
        "--data-path",
        default=str(ROOT / "tmp/PaddleScience_20260329/examples/allen_cahn/dataset/allen_cahn.mat"),
        help="Allen-Cahn reference dataset used by PaddleScience.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Training output directory. Checkpoints will be written to <output-dir>/checkpoints.",
    )
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--bottleneck-size", type=int, default=20)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--bottleneck-activation", default="tanh")
    parser.add_argument("--fourier-dim", type=int, default=256)
    parser.add_argument("--fourier-scale", type=float, default=2.0)
    parser.add_argument("--period-x", type=float, default=2.0)
    parser.add_argument("--period-x-trainable", action="store_true")
    parser.add_argument("--rwf-mean", type=float, default=1.0)
    parser.add_argument("--rwf-std", type=float, default=0.1)

    parser.add_argument("--optim", default="adam", choices=["adam", "soap"])
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--decay-steps", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--iters-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=25)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--causal-chunks", type=int, default=32)
    parser.add_argument("--causal-tol", type=float, default=1.0)
    parser.add_argument("--grad-norm-update-freq", type=int, default=1000)
    parser.add_argument("--grad-norm-momentum", type=float, default=0.9)

    parser.add_argument(
        "--official-pretrained-model",
        default=str(ROOT / "tmp/PaddleScience_20260329/examples/allen_cahn/allen_cahn_piratenet_soap_pretrained.pdparams"),
        help="Official PirateNet checkpoint used to initialize the shared body weights.",
    )
    parser.add_argument(
        "--skip-official-pretrain",
        action="store_true",
        help="Do not initialize the shared body from the official 256-width checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Resume full training state from a PaddleScience checkpoint prefix.",
    )
    parser.add_argument(
        "--freeze-body",
        action="store_true",
        help="Freeze all shared PirateNet body parameters and only train the bottleneck/output head.",
    )
    return parser.parse_args()


def plot(t_star, x_star, u_ref, u_pred, output_dir: str):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(18, 5))
    TT, XX = np.meshgrid(t_star, x_star, indexing="ij")
    u_ref = u_ref.reshape([len(t_star), len(x_star)])

    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, np.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "ac.png")
    print(f"Saving figure to {fig_path}")
    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    plt.close()


def _freeze_shared_body(model):
    trainable_prefixes = ("feature_proj", "last_fc")
    frozen = []
    trainable = []
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            param.stop_gradient = False
            trainable.append(name)
        else:
            param.stop_gradient = True
            frozen.append(name)
    print(f"Frozen shared body parameters: {len(frozen)}")
    print(f"Trainable bottleneck/head parameters: {len(trainable)}")


def _write_training_metadata(args, model, output_dir: Path):
    metadata = {
        "network_type": "bottleneck_piratenet",
        "input_keys": ["t", "x"],
        "output_keys": ["u"],
        "num_blocks": int(args.num_blocks),
        "hidden_size": int(args.hidden_size),
        "bottleneck_size": int(args.bottleneck_size),
        "activation": args.activation,
        "bottleneck_activation": args.bottleneck_activation,
        "periods": {"x": [float(args.period_x), bool(args.period_x_trainable)]},
        "fourier": {"dim": int(args.fourier_dim), "scale": float(args.fourier_scale)},
        "random_weight": {"mean": float(args.rwf_mean), "std": float(args.rwf_std)},
        "official_pretrained_model": None if args.skip_official_pretrain else str(Path(args.official_pretrained_model).resolve()),
        "feature_dim": int(getattr(model, "feature_dim", args.bottleneck_size)),
        "deepoly_factory_kwargs": {
            "input_keys": ["t", "x"],
            "output_keys": ["u"],
            "output_key": "u",
            "num_blocks": int(args.num_blocks),
            "hidden_size": int(args.hidden_size),
            "network_type": "bottleneck_piratenet",
            "bottleneck_size": int(args.bottleneck_size),
            "bottleneck_activation": args.bottleneck_activation,
            "activation": args.activation,
            "periods": {"x": [float(args.period_x), bool(args.period_x_trainable)]},
            "fourier": {"dim": int(args.fourier_dim), "scale": float(args.fourier_scale)},
            "random_weight": {"mean": float(args.rwf_mean), "std": float(args.rwf_std)},
            "device": args.device,
            "pretrained_model_path": str((output_dir / "checkpoints" / "best_model.pdparams").resolve()),
        },
    }
    metadata_path = output_dir / "deepoly_bottleneck_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote DeePoly metadata to {metadata_path}")


def main():
    args = parse_args()
    paddlesci_root = Path(args.paddlesci_root).expanduser().resolve()
    if str(paddlesci_root) not in sys.path:
        sys.path.insert(0, str(paddlesci_root))

    import paddle
    import scipy.io as sio

    import ppsci
    from ppsci.loss import mtl
    from ppsci.utils import misc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paddle.set_device(args.device)
    paddle.set_default_dtype(args.dtype)
    misc.set_random_seed(args.seed)
    dtype = np.float32 if args.dtype == "float32" else np.float64

    model = build_bottleneck_piratenet(
        input_keys=("t", "x"),
        output_keys=("u",),
        num_blocks=args.num_blocks,
        hidden_size=args.hidden_size,
        bottleneck_size=args.bottleneck_size,
        activation=args.activation,
        bottleneck_activation=args.bottleneck_activation,
        periods={"x": (args.period_x, args.period_x_trainable)},
        fourier={"dim": args.fourier_dim, "scale": args.fourier_scale},
        random_weight={"mean": args.rwf_mean, "std": args.rwf_std},
    )

    if not args.skip_official_pretrain and args.official_pretrained_model:
        load_matching_pretrained(model, args.official_pretrained_model, verbose=True)

    if args.freeze_body:
        _freeze_shared_body(model)

    equation = {"AllenCahn": ppsci.equation.AllenCahn(eps=0.01)}

    data = sio.loadmat(str(Path(args.data_path).expanduser().resolve()))
    u_ref = data["usol"].astype(dtype)
    t_star = data["t"].flatten().astype(dtype)
    x_star = data["x"].flatten().astype(dtype)

    u0 = u_ref[0, :]
    t0 = t_star[0]
    t1 = t_star[-1]
    x0 = x_star[0]
    x1 = x_star[-1]

    def gen_input_batch():
        tx = np.random.uniform([t0, x0], [t1, x1], (args.batch_size, 2)).astype(dtype)
        return {
            "t": np.sort(tx[:, 0:1], axis=0),
            "x": tx[:, 1:2],
        }

    def gen_label_batch(_input_batch):
        return {"allen_cahn": np.zeros([args.batch_size, 1], dtype)}

    pde_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": gen_input_batch,
                "label": gen_label_batch,
            },
        },
        output_expr=equation["AllenCahn"].equations,
        loss=ppsci.loss.CausalMSELoss(args.causal_chunks, "mean", tol=args.causal_tol),
        name="PDE",
    )

    ic_input = {"t": np.full([len(x_star), 1], t0, dtype=dtype), "x": x_star.reshape([-1, 1])}
    ic_label = {"u": u0.reshape([-1, 1])}
    ic = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": ic_input,
                "label": ic_label,
            },
        },
        output_expr={"u": lambda out: out["u"]},
        loss=ppsci.loss.MSELoss("mean"),
        name="IC",
    )
    constraint = {pde_constraint.name: pde_constraint, ic.name: ic}

    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        epochs=args.epochs,
        iters_per_epoch=args.iters_per_epoch,
        learning_rate=args.lr,
        gamma=args.gamma,
        decay_steps=args.decay_steps,
        by_epoch=False,
        warmup_epoch=0,
    )()
    if args.optim == "adam":
        optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)
    else:
        optimizer = ppsci.optimizer.SOAP(lr_scheduler)(model)

    tx_star = misc.cartesian_product(t_star, x_star).astype(dtype)
    eval_data = {"t": tx_star[:, 0:1], "x": tx_star[:, 1:2]}
    eval_label = {"u": u_ref.reshape([-1, 1])}
    u_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": eval_data,
                "label": eval_label,
            },
            "batch_size": args.eval_batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        {"u": lambda out: out["u"]},
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="u_validator",
    )
    validator = {u_validator.name: u_validator}

    solver = ppsci.solver.Solver(
        model,
        constraint,
        str(output_dir),
        optimizer,
        None,
        args.epochs,
        args.iters_per_epoch,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        eval_during_train=True,
        eval_freq=args.eval_freq,
        seed=args.seed,
        equation=equation,
        validator=validator,
        pretrained_model_path=None,
        checkpoint_path=args.checkpoint_path,
        eval_with_no_grad=True,
        use_tbd=False,
        loss_aggregator=mtl.GradNorm(
            model,
            len(constraint),
            args.grad_norm_update_freq,
            args.grad_norm_momentum,
        ),
    )
    solver.train()
    solver.eval()

    u_pred = solver.predict(eval_data, batch_size=args.eval_batch_size, return_numpy=True)["u"]
    u_pred = u_pred.reshape([len(t_star), len(x_star)])
    plot(t_star, x_star, u_ref, u_pred, str(output_dir))
    _write_training_metadata(args, model, output_dir)
    print(f"Best checkpoint: {output_dir / 'checkpoints' / 'best_model.pdparams'}")


if __name__ == "__main__":
    main()
