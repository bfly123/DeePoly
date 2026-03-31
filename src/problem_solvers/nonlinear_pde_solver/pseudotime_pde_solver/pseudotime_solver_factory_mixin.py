from typing import Dict, Optional

import numpy as np
from scipy.io import loadmat

from .core.fitter import PseudoTimePDEFitter
from .utils.config import PseudoTimePDEConfig
from .utils.data import PseudoTimePDEDataGenerator
from .utils.visualize import PseudoTimePDEVisualizer


class PseudoTimeSolverFactoryMixin:
    def _initialize_components(self):
        super()._initialize_components()
        self.fitter.visualizer = self.visualizer

    def _uses_reference_stable_training_schedule(self) -> bool:
        return bool(
            getattr(self.config, "steady_state_mode", True)
            and getattr(self.config, "pseudo_time_formulation", "default") == "reference_stable"
        )

    def _should_train_network(self, iteration: int) -> bool:
        """Train Spotter on the first iteration and then every configured skip interval."""
        if self._uses_reference_stable_training_schedule():
            if iteration == 0:
                if getattr(self, "_skip_initial_spotter_training", False):
                    print("  Skipping initial neural network training (reusing transferred Spotter weights)")
                    return False
                return True

            single_retrain = bool(getattr(self.config, "reference_stable_single_retrain", True))

            if single_retrain and getattr(self, "_reference_stable_post_threshold_retrain_done", False):
                print("  Skipping neural network training (legacy reference-stable schedule completed)")
                return False

            primary_update = getattr(self, "_reference_stable_last_primary_update", float("inf"))
            retrain_stop_tol = float(getattr(self.config, "reference_stable_retrain_stop_tol", 0.0))
            if retrain_stop_tol > 0.0 and primary_update <= retrain_stop_tol:
                print(
                    "  Skipping neural network training "
                    f"(reference-stable retrain disabled below {retrain_stop_tol:.2e}, "
                    f"current={primary_update:.3e})"
                )
                return False
            retrain_tol = getattr(self.config, "convergence_plateau_tol", 1e-2)
            should_train = primary_update < retrain_tol
            if should_train:
                cooldown = max(1, int(getattr(self.config, "spotter_skip", 1)))
                last_retrain_iter = getattr(self, "_reference_stable_last_retrain_iter", None)
                if last_retrain_iter is not None and iteration - last_retrain_iter < cooldown:
                    print(
                        "  Skipping neural network training "
                        f"(reference-stable cooldown {iteration - last_retrain_iter}/{cooldown})"
                    )
                    return False
                print(
                    "  Triggering legacy reference-stable Spotter retrain "
                    f"(dU_var_max={primary_update:.3e})"
                )
            else:
                print(
                    "  Skipping neural network training "
                    f"(waiting for dU_var_max < {retrain_tol:.2e}, current={primary_update:.3e})"
                )
            return should_train

        spotter_skip = getattr(self.config, "spotter_skip", 1)
        should_train = iteration % spotter_skip == 0
        if not should_train and iteration > 0:
            print(
                f"  Skipping neural network training "
                f"(spotter_skip={spotter_skip}, it={iteration})"
            )
        return should_train

    def _accept_trained_model_update(
        self,
        iter: int,
        dt: float,
        U_current: np.ndarray,
        final_loss: float,
    ) -> bool:
        if not self._uses_reference_stable_training_schedule() or iter == 0:
            return super()._accept_trained_model_update(iter, dt, U_current, final_loss)

        tolerance = getattr(self.config, "DNNtol", 1e-4)
        if final_loss <= tolerance:
            return True

        print(
            "  Rejecting legacy reference-stable Spotter retrain "
            f"(final_loss={final_loss:.3e} > DNNtol={tolerance:.3e})"
        )
        return False

    def _on_network_trained(
        self,
        iter: int,
        dt: float,
        U_current: np.ndarray,
        final_loss: float,
        *,
        accepted: bool = True,
    ) -> None:
        super()._on_network_trained(iter, dt, U_current, final_loss, accepted=accepted)
        if self._uses_reference_stable_training_schedule() and iter > 0:
            self._reference_stable_last_retrain_iter = iter
            single_retrain = bool(getattr(self.config, "reference_stable_single_retrain", True))
            self._reference_stable_post_threshold_retrain_done = accepted and single_retrain

    def _create_config(self, case_dir: str):
        return PseudoTimePDEConfig(case_dir=case_dir)

    def _create_data_generator(self):
        return PseudoTimePDEDataGenerator(self.config)

    def _create_model(self):
        if not self._needs_live_torch_model():
            return self._build_external_only_spotter_model()
        from .core.net import PseudoTimePDENet

        return PseudoTimePDENet.model_init(self.config)

    def _create_fitter(self, time_scheme: Optional[str]):
        _ = time_scheme
        return PseudoTimePDEFitter(self.config, self.data_train, self.visualizer)

    def _create_visualizer(self):
        return PseudoTimePDEVisualizer(self.config)

    def _get_default_time_scheme(self) -> Optional[str]:
        return self.DEFAULT_TIME_SCHEME

    def _process_reference_solution(
        self,
        data: Dict,
        x_ref: np.ndarray,
        mat_file: str,
    ) -> Dict:
        del data, x_ref
        mat_data = loadmat(mat_file)
        u_ref = mat_data["usol"].T.flatten()
        x_domain = self.config.x_domain
        x_ref = np.linspace(x_domain[0][0], x_domain[0][1], mat_data["x"].size)
        y_ref = np.linspace(x_domain[1][0], x_domain[1][1], mat_data["t"].size)
        return {"x_ref": x_ref, "y_ref": y_ref, "u_ref": u_ref}
