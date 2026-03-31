import warnings
from typing import Callable, Dict

from .linear_solver_backend import CUPY_AVAILABLE, cp
from .linear_solver_method_selection_mixin import LinearSolverMethodSelectionMixin
from .linear_solver_recommendation_mixin import LinearSolverRecommendationMixin
from .linear_solver_reporting_mixin import LinearSolverReportingMixin


class LinearSolverSupportMixin(
    LinearSolverRecommendationMixin,
    LinearSolverMethodSelectionMixin,
    LinearSolverReportingMixin,
):
    def _resolve_gpu_request(self, use_gpu: bool, config: object = None) -> bool:
        linear_device = getattr(config, "linear_device", "").lower() if config else ""
        if "cpu" in linear_device:
            return False

        if use_gpu and CUPY_AVAILABLE:
            try:
                if cp.cuda.runtime.getDeviceCount() == 0:
                    warnings.warn("未DetectionToCUDAEquipment，将Return退ToCPUCompute")
                    return False
            except Exception:
                warnings.warn("DetectionCUDAEquipment时Exit错，将Return退ToCPUCompute")
                return False

        return use_gpu

    def _finalize_gpu_usage(self, use_gpu: bool) -> bool:
        if use_gpu and not self.gpu_available:
            warnings.warn("请求UsingGPU但CuPyUnavailable，将Return退ToCPUCompute")
            return False
        return use_gpu and self.gpu_available

    def _build_method_registry(self) -> Dict[str, Callable]:
        methods = {
            "direct": self._solve_direct,
            "svd": self._solve_svd,
            "qr": self._solve_qr,
            "ridge": self._solve_ridge,
            "lsqr": self._solve_lsqr,
            "lsmr": self._solve_lsmr,
            "parallel_direct": self._solve_parallel_direct,
            "parallel_svd": self._solve_parallel_svd,
            "parallel_ridge": self._solve_parallel_ridge,
        }
        if self.use_gpu:
            methods.update(
                {
                    "gpu_direct": self._solve_gpu_direct,
                    "gpu_svd": self._solve_gpu_svd,
                    "gpu_qr": self._solve_gpu_qr,
                    "gpu_lstsq": self._solve_gpu_lstsq,
                    "gpu_batched_lstsq": self._solve_gpu_batched_lstsq,
                }
            )
        return methods

    def _configure_gpu_memory_pool(self) -> None:
        if not self.use_gpu:
            return

        try:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(fraction=self.gpu_memory_fraction)
            if self.verbose:
                print(f"GPUMemory usageRatioSetup为: {self.gpu_memory_fraction:.2f}")
                free_mem = self._get_available_gpu_memory()
                if free_mem:
                    print(f"AvailableGPUInner存: {free_mem:.2f} GB")
        except Exception as exc:
            warnings.warn(f"SetupGPUInner存RestrictionFail: {exc}")
