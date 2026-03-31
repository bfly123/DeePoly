from typing import Any, Dict, Optional

from .core.fitter import FixIterationPDEFitter
from .utils.config import FixIterationPDEConfig
from .utils.data import FixIterationPDEDataGenerator
from .utils.visualize import FixIterationPDEVisualizer


class FixIterationSolverFactoryMixin:
    def _create_config(self, case_dir: str):
        return FixIterationPDEConfig(case_dir=case_dir)

    def _create_data_generator(self):
        return FixIterationPDEDataGenerator(self.config)

    def _create_model(self):
        if not self._needs_live_torch_model():
            return self._build_external_only_spotter_model()
        from .core.net import FixIterationPDENet

        return FixIterationPDENet.model_init(self.config)

    def _create_fitter(self, time_scheme: Optional[str]):
        _ = time_scheme
        return FixIterationPDEFitter(config=self.config, data=self._get_iteration_dataset())

    def _create_visualizer(self):
        return FixIterationPDEVisualizer(self.config)

    def _get_default_time_scheme(self) -> Optional[str]:
        return None

    def _process_reference_solution(self, data: Dict, x_ref, mat_file: str) -> Dict:
        u_ref = data.get("usol", data.get("u", None))
        if u_ref is None:
            return None

        y_ref = None
        if "t" in data:
            y_ref = data["t"].flatten()
        elif "y" in data:
            y_ref = data["y"].flatten()

        if u_ref.ndim == 2:
            ref_dict = {
                "x_ref": x_ref.flatten(),
                "u_ref": u_ref.flatten(),
                "file_path": mat_file,
            }
            if y_ref is not None:
                ref_dict["y_ref"] = y_ref
                ref_dict["y_label"] = "t"
            return ref_dict
        return {
            "x_ref": x_ref.flatten(),
            "u_ref": u_ref.flatten(),
            "file_path": mat_file,
        }

    def _get_spotter_net_class(self):
        from ..pseudotime_pde_solver.core.spotter_init import SpotterNet

        return SpotterNet

    def _get_bc_pinn_initializer_class(self):
        from .core.bc_pinn_initializer import BCPINNInitializer

        return BCPINNInitializer
