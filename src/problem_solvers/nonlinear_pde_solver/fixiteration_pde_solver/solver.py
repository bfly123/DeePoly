import numpy as np
from typing import Dict, Any

from ..solver import NonlinearPDESolver
from .fixiteration_solver_factory_mixin import FixIterationSolverFactoryMixin
from .fixiteration_solver_output_mixin import FixIterationSolverOutputMixin
from .fixiteration_solver_runtime_mixin import FixIterationSolverRuntimeMixin


class FixIterationPDESolver(
    FixIterationSolverOutputMixin,
    FixIterationSolverRuntimeMixin,
    FixIterationSolverFactoryMixin,
    NonlinearPDESolver,
):
    pass
