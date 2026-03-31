from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


class BaseSpotterFeatureProvider:
    backend_name = "custom"
    feature_dim: Optional[int] = None

    def hidden_features(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def hidden_feature_derivatives(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        derivative: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def predict_output(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_feature_dim(
        self,
        *,
        sample_x_norm: Optional[np.ndarray] = None,
        x_min: Optional[np.ndarray] = None,
        x_max: Optional[np.ndarray] = None,
    ) -> int:
        if self.feature_dim is not None:
            return int(self.feature_dim)
        if sample_x_norm is None or x_min is None or x_max is None:
            raise ValueError("sample_x_norm, x_min, and x_max are required to infer feature_dim")
        hidden = self.hidden_features(sample_x_norm, x_min, x_max)
        self.feature_dim = int(hidden.shape[1])
        return int(self.feature_dim)


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError(
            "Torch is required for the Torch spotter backend. "
            "Install torch in the active environment or switch to a non-torch spotter backend."
        ) from exc
    return torch


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    tensor = tensor.detach().cpu()
    try:
        return tensor.numpy()
    except RuntimeError:
        return np.asarray(tensor.tolist(), dtype=np.float64)


def _default_input_transform(
    x_norm: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
) -> np.ndarray:
    return np.asarray(x_norm, dtype=np.float64) * (
        np.asarray(x_max, dtype=np.float64) - np.asarray(x_min, dtype=np.float64)
    ) + np.asarray(x_min, dtype=np.float64)


def _build_input_tensor(
    x_norm: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    device: Any,
):
    torch = _import_torch()
    x = _default_input_transform(x_norm, x_min, x_max)
    return torch.tensor(x, requires_grad=True, dtype=torch.float64, device=device)


def _compute_feature_derivatives(
    hidden: Any,
    x_tensor: Any,
    derivative: np.ndarray,
) -> np.ndarray:
    torch = _import_torch()
    jacobian = np.zeros((hidden.shape[0], hidden.shape[1]), dtype=np.float64)
    for feature_idx in range(hidden.shape[1]):
        current_grad = hidden[:, feature_idx]
        for dim, order in enumerate(derivative):
            for derivative_idx in range(int(order)):
                grad_tuple = torch.autograd.grad(
                    outputs=current_grad,
                    inputs=x_tensor,
                    grad_outputs=torch.ones_like(current_grad),
                    create_graph=True,
                    retain_graph=True,
                )
                if grad_tuple is None or grad_tuple[0] is None:
                    raise ValueError(
                        f"cannot compute {dim} derivative "
                        f"(current order {derivative_idx + 1}/{order})."
                    )
                current_grad = grad_tuple[0][:, dim : dim + 1]
        jacobian[:, feature_idx] = _tensor_to_numpy(current_grad).squeeze()
    return jacobian


class TorchSpotterFeatureProvider(BaseSpotterFeatureProvider):
    backend_name = "torch"

    def __init__(self, model: Any, device: Any = "cuda"):
        torch = _import_torch()
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.model = self.model.to(dtype=torch.float64)
        self.model.eval()
        out_layer = getattr(model, "out", None)
        in_features = getattr(out_layer, "in_features", None)
        if in_features is not None:
            self.feature_dim = int(in_features)

    def hidden_features(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ) -> np.ndarray:
        x_tensor = _build_input_tensor(x_norm, x_min, x_max, self.device)
        hidden, _ = self.model(x_tensor)
        return _tensor_to_numpy(hidden)

    def hidden_feature_derivatives(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        derivative: np.ndarray,
    ) -> np.ndarray:
        x_tensor = _build_input_tensor(x_norm, x_min, x_max, self.device)
        hidden, _ = self.model(x_tensor)
        if np.all(np.asarray(derivative, dtype=int) == 0):
            return _tensor_to_numpy(hidden)
        return _compute_feature_derivatives(hidden, x_tensor, np.asarray(derivative, dtype=int))

    def predict_output(self, points: np.ndarray) -> np.ndarray:
        torch = _import_torch()
        parameter = next(self.model.parameters(), None)
        model_dtype = parameter.dtype if parameter is not None else torch.float64
        x_tensor = torch.tensor(points, dtype=model_dtype, device=self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_tensor)
        prediction = output[-1] if isinstance(output, tuple) else output
        return _tensor_to_numpy(prediction)


class PaddleSpotterFeatureProvider(BaseSpotterFeatureProvider):
    backend_name = "paddle"

    def __init__(
        self,
        hidden_batch_fn: Callable[[Any], Any],
        output_batch_fn: Optional[Callable[[Any], Any]] = None,
        *,
        input_transform: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        device: Optional[str] = None,
        feature_dim: Optional[int] = None,
        dtype: str = "float64",
    ):
        try:
            import paddle
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Paddle is required to use PaddleSpotterFeatureProvider. "
                "Install paddlepaddle / paddlepaddle-gpu in the active environment first."
            ) from exc

        self.paddle = paddle
        self.hidden_batch_fn = hidden_batch_fn
        self.output_batch_fn = output_batch_fn
        self.input_transform = input_transform or _default_input_transform
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self.dtype = str(dtype or "float64")
        if device:
            self.paddle.set_device(str(device))

    def _build_input_tensor(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        *,
        requires_grad: bool,
    ):
        x_phys = self.input_transform(x_norm, x_min, x_max)
        np_dtype = np.float64 if self.dtype == "float64" else np.float32
        tensor = self.paddle.to_tensor(np.asarray(x_phys, dtype=np_dtype), dtype=self.dtype)
        tensor.stop_gradient = not requires_grad
        return tensor

    def _tensor_to_numpy(self, tensor) -> np.ndarray:
        return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)

    def hidden_features(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ) -> np.ndarray:
        x_tensor = self._build_input_tensor(x_norm, x_min, x_max, requires_grad=False)
        hidden = self.hidden_batch_fn(x_tensor)
        return self._tensor_to_numpy(hidden)

    def hidden_feature_derivatives(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        derivative: np.ndarray,
    ) -> np.ndarray:
        derivative = np.asarray(derivative, dtype=int)
        x_tensor = self._build_input_tensor(x_norm, x_min, x_max, requires_grad=True)
        hidden = self.hidden_batch_fn(x_tensor)
        if np.all(derivative == 0):
            return self._tensor_to_numpy(hidden)

        jacobian = np.zeros((hidden.shape[0], hidden.shape[1]), dtype=np.float64)
        for feature_idx in range(hidden.shape[1]):
            current_grad = hidden[:, feature_idx]
            for dim, order in enumerate(derivative):
                for derivative_idx in range(int(order)):
                    grad_tuple = self.paddle.grad(
                        outputs=current_grad,
                        inputs=x_tensor,
                        create_graph=True,
                        retain_graph=True,
                    )
                    if grad_tuple is None or grad_tuple[0] is None:
                        raise ValueError(
                            f"cannot compute {dim} derivative "
                            f"(current order {derivative_idx + 1}/{order})."
                        )
                    current_grad = grad_tuple[0][:, dim : dim + 1]
            jacobian[:, feature_idx] = self._tensor_to_numpy(current_grad).squeeze()
        return jacobian

    def predict_output(self, points: np.ndarray) -> np.ndarray:
        if self.output_batch_fn is None:
            raise NotImplementedError("This Paddle feature provider was created without output_batch_fn support.")
        np_dtype = np.float64 if self.dtype == "float64" else np.float32
        x_tensor = self.paddle.to_tensor(np.asarray(points, dtype=np_dtype), dtype=self.dtype)
        x_tensor.stop_gradient = True
        output = self.output_batch_fn(x_tensor)
        return self._tensor_to_numpy(output)


class JaxSpotterFeatureProvider(BaseSpotterFeatureProvider):
    backend_name = "jax"

    def __init__(
        self,
        hidden_fn: Callable[[Any], Any],
        output_fn: Optional[Callable[[Any], Any]] = None,
        *,
        input_transform: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        feature_dim: Optional[int] = None,
    ):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "JAX is required to use JaxSpotterFeatureProvider. "
                "Install jax/jaxlib in the active environment first."
            ) from exc

        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jnp
        self.hidden_fn = hidden_fn
        self.output_fn = output_fn
        self.input_transform = input_transform or self._default_input_transform
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self._hidden_batch_fn = self.jax.jit(self.jax.vmap(self.hidden_fn))
        self._output_batch_fn = (
            self.jax.jit(self.jax.vmap(self.output_fn)) if self.output_fn is not None else None
        )
        self._hidden_derivative_batch_cache: Dict[tuple[int, ...], Callable[[Any], Any]] = {}

    @staticmethod
    def _default_input_transform(
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ) -> np.ndarray:
        return _default_input_transform(x_norm, x_min, x_max)

    def _transform_points(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ):
        x_phys = self.input_transform(x_norm, x_min, x_max)
        return self.jnp.asarray(x_phys, dtype=self.jnp.float64)

    def _build_hidden_derivative_batch_fn(self, derivative: np.ndarray):
        key = tuple(int(value) for value in np.asarray(derivative, dtype=int).tolist())
        cached = self._hidden_derivative_batch_cache.get(key)
        if cached is not None:
            return cached

        def single_fn(point):
            current_fn = self.hidden_fn
            for axis, order in enumerate(key):
                for _ in range(order):
                    prev_fn = current_fn
                    current_fn = lambda p, prev_fn=prev_fn, axis=axis: self.jax.jacfwd(prev_fn)(p)[..., axis]
            return current_fn(point)

        batch_fn = self.jax.jit(self.jax.vmap(single_fn))
        self._hidden_derivative_batch_cache[key] = batch_fn
        return batch_fn

    def hidden_features(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ) -> np.ndarray:
        x_phys = self._transform_points(x_norm, x_min, x_max)
        hidden = self._hidden_batch_fn(x_phys)
        return np.asarray(self.jax.device_get(hidden), dtype=np.float64)

    def hidden_feature_derivatives(
        self,
        x_norm: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        derivative: np.ndarray,
    ) -> np.ndarray:
        derivative = np.asarray(derivative, dtype=int)
        if np.all(derivative == 0):
            return self.hidden_features(x_norm, x_min, x_max)
        x_phys = self._transform_points(x_norm, x_min, x_max)
        derivative_batch_fn = self._build_hidden_derivative_batch_fn(derivative)
        hidden = derivative_batch_fn(x_phys)
        return np.asarray(self.jax.device_get(hidden), dtype=np.float64)

    def predict_output(self, points: np.ndarray) -> np.ndarray:
        if self._output_batch_fn is None:
            raise NotImplementedError("This JAX feature provider was created without output_fn support.")
        x_phys = self.jnp.asarray(points, dtype=self.jnp.float64)
        output = self._output_batch_fn(x_phys)
        return np.asarray(self.jax.device_get(output), dtype=np.float64)


def _validate_provider(provider: Any) -> BaseSpotterFeatureProvider:
    required_methods = ("hidden_features", "hidden_feature_derivatives", "predict_output")
    missing = [method_name for method_name in required_methods if not hasattr(provider, method_name)]
    if missing:
        raise TypeError(
            "Spotter feature provider is missing required methods: "
            + ", ".join(sorted(missing))
        )
    return provider


def _load_factory_from_file(
    factory_file: str,
    factory_function: str,
) -> Callable[..., Any]:
    path = Path(factory_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Spotter feature factory file not found: {path}")

    spec = importlib.util.spec_from_file_location("deepoly_spotter_feature_factory", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spotter feature factory module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    factory = getattr(module, factory_function, None)
    if not callable(factory):
        raise AttributeError(f"Factory function '{factory_function}' not found in: {path}")
    return factory


def create_spotter_feature_provider(
    config,
    model: Any,
    *,
    data_train: Optional[Dict[str, Any]] = None,
    data_test: Optional[Dict[str, Any]] = None,
) -> BaseSpotterFeatureProvider:
    return create_spotter_feature_provider_from_config(
        config=config,
        model=model,
        data_train=data_train,
        data_test=data_test,
    )


def create_spotter_feature_provider_from_config(
    *,
    config,
    model: Any,
    data_train: Optional[Dict[str, Any]] = None,
    data_test: Optional[Dict[str, Any]] = None,
) -> BaseSpotterFeatureProvider:
    backend = str(getattr(config, "spotter_feature_backend", "torch") or "torch").lower()
    provider_cfg = getattr(config, "spotter_feature", {}) or {}
    return create_spotter_feature_provider_from_settings(
        backend=backend,
        provider_cfg=provider_cfg,
        case_dir=getattr(config, "case_dir", None),
        config=config,
        model=model,
        data_train=data_train,
        data_test=data_test,
        device=getattr(config, "device", "cuda"),
    )


def create_spotter_feature_provider_from_settings(
    *,
    backend: str,
    provider_cfg: Optional[Dict[str, Any]] = None,
    case_dir: Optional[str] = None,
    config: Optional[Any] = None,
    model: Any = None,
    data_train: Optional[Dict[str, Any]] = None,
    data_test: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
) -> BaseSpotterFeatureProvider:
    backend = str(backend or "torch").lower()
    provider_cfg = dict(provider_cfg or {})
    runtime_config = config or SimpleNamespace(case_dir=case_dir, device=device, n_dim=None)

    if backend == "torch":
        if model is None:
            raise ValueError("Torch spotter backend requires a live torch model instance.")
        return TorchSpotterFeatureProvider(model=model, device=device)

    external_provider_classes = {
        "jax": JaxSpotterFeatureProvider,
        "paddle": PaddleSpotterFeatureProvider,
    }
    provider_class = external_provider_classes.get(backend)
    if provider_class is None:
        raise ValueError(f"Unsupported spotter_feature_backend: {backend}")

    factory_file = provider_cfg.get("factory_file")
    factory_function = provider_cfg.get("factory_function", "build_provider")
    factory_kwargs = dict(provider_cfg.get("factory_kwargs", {}))

    if not factory_file:
        raise ValueError(
            f"{backend.upper()} feature backend requires 'spotter_feature.factory_file' "
            "to build the external provider."
        )

    if not Path(factory_file).is_absolute():
        case_root = Path(case_dir or ".").expanduser().resolve()
        factory_file = str((case_root / factory_file).resolve())

    factory = _load_factory_from_file(factory_file, factory_function)
    provider = factory(
        config=runtime_config,
        model=model,
        case_dir=case_dir,
        data_train=data_train,
        data_test=data_test,
        provider_class=provider_class,
        **factory_kwargs,
    )
    return _validate_provider(provider)


def resolve_spotter_feature_dim(
    feature_source,
    *,
    sample_x_norm: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    device: str = "cpu",
) -> int:
    if hasattr(feature_source, "get_feature_dim"):
        return int(
            feature_source.get_feature_dim(
                sample_x_norm=np.asarray(sample_x_norm, dtype=np.float64),
                x_min=np.asarray(x_min, dtype=np.float64),
                x_max=np.asarray(x_max, dtype=np.float64),
            )
        )

    torch = _import_torch()
    x_tensor = _build_input_tensor(
        np.asarray(sample_x_norm, dtype=np.float64),
        np.asarray(x_min, dtype=np.float64),
        np.asarray(x_max, dtype=np.float64),
        device,
    )
    output = feature_source(x_tensor)
    hidden = output[0] if isinstance(output, tuple) else output
    feature_dim = int(hidden.shape[1])
    if hasattr(feature_source, "feature_dim"):
        feature_source.feature_dim = feature_dim
    return feature_dim
