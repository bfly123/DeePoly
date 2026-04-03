from .spotter_equation_processor import EquationProcessor
from src.abstract_class.config.config_io import resolve_case_config_path
from .spotter_generator import (
    AutoCodeGenerator,
    update_hybrid_fitter_code,
    update_physics_loss_from_config,
    update_pytorch_net_code,
)

__all__ = [
    "EquationProcessor",
    "AutoCodeGenerator",
    "update_hybrid_fitter_code",
    "update_physics_loss_from_config",
    "update_pytorch_net_code",
]


if __name__ == "__main__":
    config_path = resolve_case_config_path("cases/linear_pde_cases/poisson_2d_sinpixsinpiy")
    try:
        update_physics_loss_from_config(config_path)
        print("代yardGenerateSuccess!")
    except Exception as exc:
        print(f"代yardGenerateFail: {exc}")
