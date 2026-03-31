import numpy as np


class TimeConfigSchemeMixin:
    def _setup_time_scheme_params(self):
        print(f"Time scheme: {self.time_scheme}")
        defaults = self._time_scheme_defaults()
        for key, value in defaults.items():
            setattr(self, key, value)
        for key, value in self.time_scheme_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        print(
            "Time scheme parameters: "
            f"{{'gamma': {getattr(self, 'gamma', 'N/A')}, "
            f"'max_time_steps': {self.max_time_steps}, "
            f"'time_output_interval': {self.time_output_interval}, "
            f"'adaptive_dt': {self.adaptive_dt}, "
            f"'cfl_number': {getattr(self, 'cfl_number', 'N/A')}, "
            f"'stability_check': {self.stability_check}, "
            f"'max_eigenvalue_estimate': {getattr(self, 'max_eigenvalue_estimate', 'N/A')}}}"
        )

    def _time_scheme_defaults(self):
        if self.time_scheme in ["IMEX_RK_2_2_2", "IMEX_222"]:
            return {
                "gamma": 1 - 1 / np.sqrt(2),
                "max_time_steps": 1000,
                "time_output_interval": 10,
                "adaptive_dt": False,
                "cfl_number": 0.5,
                "stability_check": True,
                "max_eigenvalue_estimate": 100.0,
            }
        if self.time_scheme == "onestep_predictor":
            return {
                "max_time_steps": int(self.T / self.dt) + 1,
                "time_output_interval": 1,
                "adaptive_dt": False,
                "stability_check": False,
            }
        return {
            "max_time_steps": 1000,
            "time_output_interval": 10,
            "adaptive_dt": False,
            "cfl_number": 0.5,
            "stability_check": True,
            "max_eigenvalue_estimate": 100.0,
        }
