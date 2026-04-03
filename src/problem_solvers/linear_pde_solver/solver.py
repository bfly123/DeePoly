from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import os
import json
import importlib.util
import argparse
import time
import sys

from src.abstract_class.config.config_io import resolve_case_config_path
from src.spotter_runtime import create_spotter_session

from .core import LinearPDENet, LinearPDEFitter
from .utils import LinearPDEConfig, LinearPDEDataGenerator, LinearPDEVisualizer


class LinearPDESolver:

    def __init__(self, config=None, case_dir=None):
        self.case_dir = case_dir
        self.config = config
        self.datagen = LinearPDEDataGenerator(self.config)
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")
        self.visualizer = LinearPDEVisualizer(self.config)

        self.model = LinearPDENet(self.config).to(self.config.device)
        self.spotter_session = create_spotter_session(
            config=self.config,
            model=self.model,
            data_train=self.data_train,
            data_test=self.data_test,
        )

        self.output_module = self._load_output_module()

    def _load_output_module(self):
        if not self.case_dir:
            raise ValueError("Case path not set, cannot load output.py module")

        output_path = os.path.join(self.case_dir, "output.py")
        if not os.path.exists(output_path):
            raise ValueError(f"Required output.py module not found: {output_path}")

        try:
            print(f"Loading custom output and visualization module: {output_path}")
            if self.case_dir not in sys.path:
                sys.path.insert(0, self.case_dir)
            spec = importlib.util.spec_from_file_location("custom_output_module", output_path)
            output_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(output_module)
            return output_module
        except Exception as e:
            raise RuntimeError(f"Error loading output.py module: {e}")

    def _get_spotter_feature_source(self):
        session = getattr(self, "spotter_session", None)
        if session is None:
            return self.model
        feature_source = session.feature_source()
        return feature_source if feature_source is not None else self.model

    def _train_spotter(self) -> float:
        feature_source = self._get_spotter_feature_source()
        train_external = getattr(feature_source, "train_external", None)
        allow_local_torch = bool(getattr(self.spotter_session, "allow_local_torch_training", lambda: True)())
        if callable(train_external) and not allow_local_torch:
            print("Starting external spotter training...")
            result = train_external({"problem_type": "linear_pde"})
            final_loss = float(result.get("final_loss", float("nan")))
            print(f"External spotter training completed, final loss: {final_loss:.8e}")
            return final_loss

        data_gpu = self.model.prepare_gpu_data(self.data_train)
        print("Starting neural network training...")
        self.model.eval()
        self.model = self.model.train_net(self.data_train, self.model, data_gpu)
        final_loss = self.model.physics_loss(data_gpu).item()
        print(f"Neural network training completed, final loss: {final_loss:.8e}")
        return float(final_loss)

    def _solve_with_pinns_only(self, *, feature_source, result_dir: str, start_time: float):
        total_time = time.time() - start_time
        scoper_time = total_time
        sniper_time = 0.0

        print("PINNs-only mode detected: skipping Snipper fitting and coefficient solve.")
        print(f"Total solution time: {total_time:.2f} seconds")
        print(f"Scoper time: {scoper_time:.2f} seconds")
        print(f"Sniper time: {sniper_time:.2f} seconds")

        self.output_module.generate_output(
            self.config,
            self.data_train,
            self.data_test,
            None,
            feature_source,
            None,
            result_dir,
            self.visualizer,
            total_time,
            scoper_time,
            sniper_time,
        )

    def solve(self, result_dir=None):
        print("Starting to solve the linear partial differential equation problem...")

        if result_dir is None:
            result_dir = os.path.join(self.case_dir, "results")
        os.makedirs(result_dir, exist_ok=True)
        export_effective = getattr(self.config, "export_effective_config", None)
        if callable(export_effective):
            export_effective(
                result_dir,
                extra_metadata={"solver_family": "linear_pde"},
            )
        print(f"Results will be saved in: {os.path.abspath(result_dir)}")

        print("Initializing fitter...")
        start_time = time.time()
        final_loss = self._train_spotter()
        scoper_time = time.time() - start_time

        feature_source = self._get_spotter_feature_source()
        if self.config.uses_pinns_only():
            self.fitter = None
            self._solve_with_pinns_only(
                feature_source=feature_source,
                result_dir=result_dir,
                start_time=start_time,
            )
            return

        self.fitter = LinearPDEFitter(self.config, self.data_train)
        self.fitter.fitter_init(feature_source)

        print("Starting equation fitting...")
        coeffs = self.fitter.fit()
        total_time = time.time() - start_time
        sniper_time = total_time - scoper_time

        print(f"Total solution time: {total_time:.2f} seconds")
        print(f"Scoper time: {scoper_time:.2f} seconds")
        print(f"Sniper time: {sniper_time:.2f} seconds")

        self.output_module.generate_output(
            self.config,
            self.data_train,
            self.data_test,
            self.fitter,
            feature_source,
            coeffs,
            result_dir,
            self.visualizer,
            total_time,
            scoper_time,
            sniper_time,
        )


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    case_dir = "cases/linear_pde_cases/test_poisson_2d"
    print(f"Using specified case path: {case_dir}")

    try:
        resolve_case_config_path(case_dir)
        original_dir = os.getcwd()
        try:
            solver = LinearPDESolver(case_path=case_dir)
            solver.solve()
        finally:
            os.chdir(original_dir)
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc


if __name__ == "__main__":
    package_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(package_path, "../../..")))
    __package__ = "src.problem_solvers.linear_pde_solver"

    main()
