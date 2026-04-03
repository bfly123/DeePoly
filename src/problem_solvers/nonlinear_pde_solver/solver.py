"""
NonlinearPDESolver - 统一的非线性PDE求解器基类

基于统一迭代框架设计，支持三种求解策略：
1. TimePDESolver - 物理时间演化
2. GeneralPDESolver - 伪时间步迭代至稳态
3. FixedPointSolver - 不动点迭代（无时间步）

核心思想：所有求解都是迭代 U^{n+1} = F(U^n, state)
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, Callable, List
from dataclasses import dataclass, field
import copy
import numpy as np
import scipy.io
import os

from src.abstract_class.config.config_io import resolve_case_config_path
from src.spotter_runtime import NullSpotterModel, create_spotter_session


def _import_torch():
    import torch

    return torch


@dataclass
class IterationState:
    """迭代状态 - 统一的状态管理

    用于管理三种不同求解策略的迭代状态：
    - TimePDE: 使用 iter, time, dt
    - GeneralPDE: 使用 iter, time (τ), dt (dτ), residual, converged
    - FixedPoint: 使用 iter, residual, converged
    """
    # 共性状态：迭代次数
    iter: int = 0

    # 时间相关（TimePDE, GeneralPDE）
    time: float = 0.0                # 当前时间（T 或 τ）
    dt: float = 0.0                  # 时间步长

    # 收敛相关（GeneralPDE, FixedPoint）
    residual: float = float('inf')   # 残差
    converged: bool = False          # 是否收敛

    # 解相关
    U_prev: Optional[np.ndarray] = None  # 上一步的解
    coeffs: Optional[np.ndarray] = None  # 当前系数

    # 其他自定义状态
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """确保 custom 字典已初始化"""
        if self.custom is None:
            self.custom = {}


class NonlinearPDESolver(ABC):
    """非线性PDE求解器统一基类 - 基于统一迭代框架

    核心思想：所有求解策略都是迭代 U^{n+1} = F(U^n, state)

    区别仅在于：
    1. 终止条件：时间达到 vs 残差收敛
    2. 迭代步骤：time_step vs fit
    3. 状态更新：时间递增 vs 残差计算

    代码复用率：~80%
    """

    # ==================== 统一迭代框架（核心方法） ====================

    def _unified_iteration_loop(
        self,
        U_init: np.ndarray,
        initial_state: IterationState,
        iteration_step: Callable[[np.ndarray, IterationState], Tuple[np.ndarray, IterationState]],
        termination_check: Callable[[IterationState], bool],
        state_update: Optional[Callable[[np.ndarray, np.ndarray, IterationState], IterationState]] = None,
        print_progress: Optional[Callable[[IterationState], None]] = None
    ) -> Tuple[np.ndarray, IterationState]:
        """统一的迭代循环框架 - 三种 Solver 的共同抽象

        这是最核心的方法，提供了一个通用的迭代框架。

        Args:
            U_init: 初始解 (n_points, n_eqs)
            initial_state: 初始迭代状态
            iteration_step: 迭代步骤函数
                签名：(U, state) -> (U_new, state_new)
                - TimePDE: time_step(U, state) -> (U_new, state)
                - GeneralPDE: pseudo_time_step(U, state) -> (U_new, state)
                - FixedPoint: fixed_point_step(U, state) -> (U_new, state)

            termination_check: 终止条件检查函数
                签名：(state) -> bool
                - TimePDE: lambda s: s.time >= T_final
                - GeneralPDE: lambda s: s.converged or s.iter >= max_iter
                - FixedPoint: lambda s: s.converged or s.iter >= max_iter

            state_update: 状态更新函数（可选）
                签名：(U_old, U_new, state) -> state_new
                用于自定义状态更新逻辑（如自适应时间步）

            print_progress: 进度打印函数（可选）
                签名：(state) -> None

        Returns:
            (U_final, final_state): 最终解和最终状态
        """
        U = U_init.copy()
        state = initial_state

        while not termination_check(state):
            # 执行一步迭代
            U_new, state_new = iteration_step(U, state)

            # 可选：自定义状态更新（如自适应步长）
            if state_update is not None:
                state_new = state_update(U, U_new, state_new)

            # 更新解和状态
            U = U_new
            state = state_new

            # 可选：打印进度
            if print_progress is not None:
                print_progress(state)

        return U, state

    # ==================== 辅助方法（具体方法 - 100%共性） ====================

    def _compute_residual(self, U_new: np.ndarray, U_old: np.ndarray) -> float:
        """计算相对残差 - 标准实现

        用于 GeneralPDE 和 FixedPoint 的收敛判断

        Args:
            U_new: 新解
            U_old: 旧解

        Returns:
            相对残差 ||U_new - U_old|| / ||U_old||
        """
        return np.linalg.norm(U_new - U_old) / (np.linalg.norm(U_old) + 1e-10)

    def _update_convergence_state(
        self,
        U_old: np.ndarray,
        U_new: np.ndarray,
        state: IterationState,
        conv_tol: float
    ) -> IterationState:
        """更新收敛状态 - 标准实现

        用于 GeneralPDE 和 FixedPoint

        Args:
            U_old: 旧解
            U_new: 新解
            state: 当前状态
            conv_tol: 收敛容差

        Returns:
            更新后的状态
        """
        residual = self._compute_residual(U_new, U_old)
        state.residual = residual
        state.converged = (residual < conv_tol)
        state.U_prev = U_old.copy()
        state.iter += 1
        return state

    def _update_time_state(
        self,
        state: IterationState,
        dt: Optional[float] = None
    ) -> IterationState:
        """更新时间状态 - 标准实现

        用于 TimePDE 和 GeneralPDE

        Args:
            state: 当前状态
            dt: 时间步长（如果为 None，使用 state.dt）

        Returns:
            更新后的状态
        """
        if dt is None:
            dt = state.dt
        state.time += dt
        state.dt = dt
        state.iter += 1
        return state

    def _train_neural_network_step(
        self,
        iter: int,
        dt: float,
        U_current: np.ndarray,
        force: bool = False
    ):
        """神经网络训练步骤 - 通用框架

        根据配置决定是否训练神经网络，并更新 fitter

        Args:
            iter: 当前迭代次数
            dt: 当前时间步长
            U_current: 当前解
            force: 是否强制训练（忽略 spotter_skip）
        """
        # 记录最近一次训练损失（用于子类钩子）
        self._last_training_loss: Optional[float] = None

        # 检查是否需要跳过神经网络训练
        if not force and not self._should_train_network(iter):
            # 仍然需要重新初始化 fitter，但使用现有模型参数
            self._sync_spotter_session(model_updated=False)
            self.fitter.fitter_init(self._get_spotter_feature_source())
            return

        if self._uses_external_spotter_feature_backend() and not self._allow_torch_training_with_external_features():
            if self._can_train_external_spotter_backend():
                backend_name = self._get_active_spotter_backend_name()
                print(f"  Training External Spotter backend ({backend_name})...")
                result = self._train_external_spotter_step(iter=iter, dt=dt, U_current=U_current)
                final_loss_value = result.get("final_loss") if isinstance(result, dict) else None
                if final_loss_value is not None:
                    final_loss_value = float(final_loss_value)
                    print(f"  ✓ External Training Completed - Final Loss: {final_loss_value:.6e}")
                self._last_training_loss = final_loss_value
                self._sync_spotter_session(model_updated=False)
                self.fitter.fitter_init(self._get_spotter_feature_source())
                self._on_network_trained(
                    iter,
                    dt,
                    U_current,
                    float(final_loss_value) if final_loss_value is not None else float("nan"),
                    accepted=True,
                )
                return
            print(
                "  Skipping Torch Spotter training "
                f"(spotter_feature_backend={self._get_active_spotter_backend_name()})"
            )
            self._sync_spotter_session(model_updated=False)
            self.fitter.fitter_init(self._get_spotter_feature_source())
            return

        print(f"  Training Neural Network...")

        # 准备 GPU 数据
        data_GPU = self.model.prepare_gpu_data(self.data_train, U_current)
        previous_state = copy.deepcopy(self.model.state_dict())

        # 训练模式
        self.model.train()

        # 执行训练
        self.model.train_net(self.data_train, self.model, data_GPU, dt=dt)

        # 评估模式并计算最终损失
        self.model.eval()
        final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
        print(f"  ✓ Training Completed - Final Loss: {final_loss:.6e}")
        accepted = self._accept_trained_model_update(iter, dt, U_current, final_loss)

        if not accepted:
            self.model.load_state_dict(previous_state)
            self.model.eval()
            print("  Rejected Spotter update; restored previous model parameters")

        # 记录最后一次训练损失，供子类使用
        self._last_training_loss = final_loss

        self._sync_spotter_session(model_updated=accepted)
        # 重新初始化 fitter 以使用更新后的模型
        self.fitter.fitter_init(self._get_spotter_feature_source())

        # 触发子类钩子以便执行额外的后处理（例如记录、可视化）
        self._on_network_trained(iter, dt, U_current, final_loss, accepted=accepted)

    def _should_train_network(self, iter: int) -> bool:
        """判断是否需要训练网络 - 可覆盖

        Args:
            iter: 当前迭代次数

        Returns:
            是否需要训练
        """
        spotter_skip = getattr(self.config, 'spotter_skip', 1)
        return (iter % spotter_skip == 0)

    # ==================== 初始化方法（100%共性） ====================

    def __init__(self, config: Optional[Any] = None, case_dir: Optional[str] = None) -> None:
        """统一的初始化流程

        Args:
            config: 配置对象（可选）
            case_dir: 案例目录路径（可选）
        """
        self._initialize_config(config, case_dir)
        self._initialize_components()
        self._solver_specific_init()

    def _initialize_config(self, config: Optional[Any], case_dir: Optional[str]) -> None:
        """配置初始化 - 100%相同

        Args:
            config: 配置对象
            case_dir: 案例目录
        """
        if config is not None:
            self.config = config
        elif case_dir is not None:
            self.config = self._create_config(case_dir)
        else:
            raise ValueError("Either config object or case_dir must be provided")

        self.case_dir = case_dir if case_dir else getattr(self.config, "case_dir", None)

        # 导出配置文件 - 已禁用，避免覆盖原始config.json
        # if hasattr(self.config, "export_to_json"):
        #     self.config.export_to_json("config.json")

    def _initialize_components(self) -> None:
        """组件初始化 - 100%相同

        创建并初始化所有核心组件：
        - 数据生成器
        - 训练和测试数据
        - 神经网络模型
        - 可视化器
        - Fitter
        - 参考解
        """
        # 数据生成器
        self.datagen = self._create_data_generator()

        # 生成训练和测试数据
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")
        self.data_snipper = self._build_snipper_dataset()

        # 神经网络模型
        self.model = self._create_model()
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.config.device)

        # 可视化器
        self.visualizer = self._create_visualizer()

        # Fitter（使用默认或配置的时间格式）
        time_scheme = getattr(self.config, 'time_scheme', self._get_default_time_scheme())
        self.fitter = self._create_fitter(time_scheme)

        # 加载参考解（如果指定）
        self.reference_solution = self._load_reference_solution()

        self.spotter_session = self._create_spotter_session()
        self.spotter_orchestrator = getattr(self.spotter_session, "orchestrator", None)
        self.spotter_feature_provider = getattr(self.spotter_session, "provider", None)

        # 创建结果目录
        os.makedirs(self.config.results_dir, exist_ok=True)
        export_effective = getattr(self.config, "export_effective_config", None)
        if callable(export_effective):
            export_effective(
                self.config.results_dir,
                extra_metadata={"solver_family": "nonlinear_pde"},
            )
        self._save_spotter_selection_report()

    def _build_snipper_dataset(self):
        sampling_cfg = getattr(self.config, "snipper_sampling", {}) or {}
        if not isinstance(sampling_cfg, dict) or not sampling_cfg:
            return None
        if not bool(sampling_cfg.get("enabled", True)):
            return None

        snipper_config = copy.deepcopy(self.config)
        if "points_domain" in sampling_cfg:
            snipper_config.points_domain = sampling_cfg["points_domain"]
        if "points_per_swap" in sampling_cfg:
            snipper_config.points_per_swap = sampling_cfg["points_per_swap"]

        boundary_points = sampling_cfg.get("boundary_points")
        boundary_scale = sampling_cfg.get("boundary_points_scale")
        if hasattr(snipper_config, "boundary_conditions") and isinstance(snipper_config.boundary_conditions, list):
            boundary_conditions = copy.deepcopy(snipper_config.boundary_conditions)
            for bc in boundary_conditions:
                if "points" not in bc:
                    continue
                if boundary_points is not None:
                    bc["points"] = int(boundary_points)
                elif boundary_scale is not None:
                    bc["points"] = max(1, int(round(float(bc["points"]) * float(boundary_scale))))
            snipper_config.boundary_conditions = boundary_conditions

        snipper_generator = self.datagen.__class__(snipper_config)
        snipper_data = snipper_generator.generate_data("train")
        print(
            "Using separate Snipper sampling dataset: "
            f"points_domain={getattr(snipper_config, 'points_domain', 'N/A')}, "
            f"points_per_swap={getattr(snipper_config, 'points_per_swap', 'N/A')}, "
            f"n_points={snipper_data['x'].shape[0]}"
        )
        return snipper_data

    def _get_iteration_dataset(self):
        return getattr(self, "data_snipper", None) or self.data_train

    def _load_reference_solution(self) -> Optional[Dict]:
        """加载参考解 - 85%相同，具体处理由子类实现

        Returns:
            参考解字典，如果不存在则返回 None
        """
        if not hasattr(self.config, 'reference_solution') or not self.config.reference_solution:
            return None

        ref_path = self.config.reference_solution

        # 如果是相对路径，转为绝对路径
        if not os.path.isabs(ref_path):
            case_dir = getattr(self.config, 'case_dir', os.getcwd())
            ref_path = os.path.join(case_dir, ref_path)

        # 检查 .mat 文件
        if ref_path.endswith('.m'):
            mat_file = ref_path.replace('.m', '.mat')
        else:
            mat_file = ref_path

        if not os.path.exists(mat_file):
            print(f"Warning: Reference solution file not found: {mat_file}")
            return None

        try:
            print(f"Loading reference solution: {mat_file}")
            data = scipy.io.loadmat(mat_file)

            # 提取 x 坐标（所有求解器都需要）
            if 'x' in data:
                x_ref = data['x'].flatten()
            else:
                print("Warning: 'x' not found in reference solution file")
                return None

            # 委托给子类处理具体数据
            return self._process_reference_solution(data, x_ref, mat_file)

        except Exception as e:
            print(f"Error: Cannot read reference solution file {mat_file}: {e}")
            return None

    def _to_numpy_cpu(self, data):
        """GPU数据转CPU numpy数组 - 工具方法

        Args:
            data: 输入数据（torch.Tensor 或 np.ndarray）

        Returns:
            numpy 数组，如果转换失败则返回原始数据
        """
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return np.asarray(data)
        try:
            torch = _import_torch()
        except ImportError:
            torch = None
        if torch is not None and isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        try:
            return np.asarray(data)
        except:
            return data

    def _interpolate_to_test_grid(self, U_train: np.ndarray, data_source: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """从训练点插值到测试网格 - 工具方法

        Args:
            U_train: 训练点上的解 (n_train, n_eqs)

        Returns:
            测试点上的解 (n_test, n_eqs)
        """
        from scipy.interpolate import griddata

        data_source = data_source or self.data_train
        x_train = data_source['x']
        x_test = self.data_test['x']
        n_eqs = U_train.shape[1] if U_train.ndim > 1 else 1

        if n_eqs == 1:
            U_train = U_train.reshape(-1, 1)

        U_test = np.zeros((x_test.shape[0], n_eqs))

        for eq_idx in range(n_eqs):
            # 线性插值
            result = griddata(
                x_train, U_train[:, eq_idx], x_test,
                method='linear', fill_value=np.nan
            )
            U_test[:, eq_idx] = result.flatten() if result.ndim > 1 else result

            # 对 NaN 值使用最近邻插值
            nan_mask = np.isnan(U_test[:, eq_idx])
            if np.any(nan_mask):
                result_nn = griddata(
                    x_train, U_train[:, eq_idx], x_test[nan_mask],
                    method='nearest'
                )
                U_test[nan_mask, eq_idx] = result_nn.flatten() if result_nn.ndim > 1 else result_nn

        print(f"  Interpolated from {x_train.shape[0]} training points to {x_test.shape[0]} test points")
        return U_test

    def _normalize_points_per_dim(self, points_per_dim=None) -> List[int]:
        """Normalize grid resolution config to a per-dimension integer list."""
        if points_per_dim is None:
            points_per_dim = getattr(self.config, "points_domain_test", 200)
        if isinstance(points_per_dim, int):
            return [int(points_per_dim)] * self.config.n_dim
        values = [int(value) for value in points_per_dim]
        if len(values) != self.config.n_dim:
            raise ValueError(
                f"Expected {self.config.n_dim} grid dimensions, got {len(values)}: {points_per_dim}"
            )
        return values

    def _generate_structured_evaluation_points(
        self,
        points_per_dim=None,
        *,
        include_boundary: bool = True,
    ) -> np.ndarray:
        """Build a structured evaluation grid for diagnostics/visualization."""
        counts = self._normalize_points_per_dim(points_per_dim)
        grids = []
        for dim, count in enumerate(counts):
            lower = float(self.config.x_domain[dim][0])
            upper = float(self.config.x_domain[dim][1])
            if include_boundary:
                grids.append(np.linspace(lower, upper, count, dtype=np.float64))
                continue
            if count <= 1:
                grids.append(np.array([(lower + upper) * 0.5], dtype=np.float64))
                continue
            delta = (upper - lower) / (count + 1)
            grids.append(np.linspace(lower + delta, upper - delta, count, dtype=np.float64))

        indexing = "xy" if self.config.n_dim == 2 else "ij"
        mesh = np.meshgrid(*grids, indexing=indexing)
        return np.stack(mesh, axis=-1).reshape(-1, self.config.n_dim)

    def _segment_multi_index(self, segment_idx: int) -> List[int]:
        """Return the per-dimension segment index with the first dimension varying fastest."""
        indices = []
        remainder = int(segment_idx)
        for dim_count in self.config.n_segments:
            indices.append(remainder % dim_count)
            remainder //= dim_count
        return indices

    def _create_unique_segment_mask(self, x: np.ndarray, segment_idx: int) -> np.ndarray:
        """Create a unique ownership mask using lower-open / upper-closed interior segments."""
        x_min = self.config.x_min[segment_idx]
        x_max = self.config.x_max[segment_idx]
        seg_indices = self._segment_multi_index(segment_idx)
        tol = 1e-12
        masks = []
        for dim, seg_idx_dim in enumerate(seg_indices):
            lower = x_min[dim]
            upper = x_max[dim]
            if seg_idx_dim == 0:
                masks.append(x[:, dim] >= lower - tol)
            else:
                masks.append(x[:, dim] > lower + tol)
            masks.append(x[:, dim] <= upper + tol)
        return np.logical_and.reduce(masks)

    def _split_points_unique_to_segments(self, x_global: np.ndarray) -> List[np.ndarray]:
        """Split a structured grid into segments with unique ownership for interface points."""
        x_global = np.asarray(x_global, dtype=np.float64)
        segments = []
        assigned = np.zeros(x_global.shape[0], dtype=bool)
        for segment_idx in range(self.Ns if hasattr(self, "Ns") else int(np.prod(self.config.n_segments))):
            mask = self._create_unique_segment_mask(x_global, segment_idx)
            overlap = mask & assigned
            if np.any(overlap):
                raise RuntimeError(
                    f"Duplicate ownership detected while building diagnostic grid for segment {segment_idx}"
                )
            assigned |= mask
            segments.append(x_global[mask])
        if not np.all(assigned):
            missing = int(np.count_nonzero(~assigned))
            raise RuntimeError(f"{missing} diagnostic points were not assigned to any segment")
        return segments

    def _build_direct_evaluation_dataset(
        self,
        points_per_dim=None,
        *,
        include_boundary: bool = True,
    ) -> Dict[str, Any]:
        """Build a structured dataset for direct Snipper/Spotter reconstruction."""
        x_global = self._generate_structured_evaluation_points(
            points_per_dim,
            include_boundary=include_boundary,
        )
        x_segments = self._split_points_unique_to_segments(x_global)
        x_segments_norm = [
            self.datagen._normalize_data(
                x_segments[segment_idx],
                self.config.x_min[segment_idx],
                self.config.x_max[segment_idx],
            )
            for segment_idx in range(len(x_segments))
        ]
        return {
            "x": np.vstack(x_segments),
            "x_segments": x_segments,
            "x_segments_norm": x_segments_norm,
        }

    def _predict_network_on_points(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the Spotter network directly on arbitrary point clouds."""
        feature_source = self._get_spotter_feature_source()
        if hasattr(feature_source, "predict_output"):
            return np.asarray(feature_source.predict_output(points), dtype=np.float64)
        torch = _import_torch()
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(points, dtype=torch.float64, device=self.config.device)
            _, U_pred = self.model(x_tensor)
        return U_pred.detach().cpu().numpy()

    def _build_flow_diagnostic_results(self, coeffs: np.ndarray) -> Optional[Dict[str, Any]]:
        """Build direct reconstruction payloads for 2D flow diagnostics."""
        if getattr(self.config, "n_dim", 0) != 2 or getattr(self.config, "n_eqs", 0) < 2:
            return None
        try:
            diagnostic_data = self._build_direct_evaluation_dataset(include_boundary=True)
            sniper_prediction, _ = self.fitter.construct(
                diagnostic_data,
                self._get_spotter_feature_source(),
                self._to_numpy_cpu(coeffs),
            )
            spotter_prediction = self._predict_network_on_points(diagnostic_data["x"])
            return {
                "data": diagnostic_data,
                "predictions": {
                    "sniper": sniper_prediction,
                    "spotter": spotter_prediction,
                },
            }
        except Exception as exc:
            print(f"  Warning: failed to build direct flow diagnostics: {exc}")
            return None

    # ==================== 抽象方法（子类必须实现） ====================

    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """主求解方法 - 子类使用 _unified_iteration_loop 实现

        子类应该：
        1. 定义 iteration_step 函数
        2. 定义 termination_check 函数
        3. 可选定义 state_update 和 print_progress 函数
        4. 调用 _unified_iteration_loop
        5. 返回结果字典
        """
        pass

    @abstractmethod
    def _create_config(self, case_dir: str):
        """创建配置对象 - 工厂方法

        Args:
            case_dir: 案例目录

        Returns:
            配置对象
        """
        pass

    @abstractmethod
    def _create_data_generator(self):
        """创建数据生成器 - 工厂方法

        Returns:
            数据生成器对象
        """
        pass

    @abstractmethod
    def _create_model(self):
        """创建神经网络模型 - 工厂方法

        Returns:
            模型对象
        """
        pass

    @abstractmethod
    def _create_fitter(self, time_scheme: Optional[str]):
        """创建Fitter对象 - 工厂方法

        Args:
            time_scheme: 时间格式（可选）

        Returns:
            Fitter 对象
        """
        pass

    @abstractmethod
    def _create_visualizer(self):
        """创建可视化器 - 工厂方法

        Returns:
            可视化器对象
        """
        pass

    @abstractmethod
    def _get_default_time_scheme(self) -> Optional[str]:
        """获取默认时间格式 - 工厂方法

        Returns:
            默认时间格式字符串，如果不适用则返回 None
        """
        pass

    @abstractmethod
    def _process_reference_solution(self, data: Dict, x_ref: np.ndarray, mat_file: str) -> Dict:
        """处理参考解数据 - 维度特定处理

        不同维度的 PDE 有不同的数据结构：
        - 1D+时间: (x, t, u)
        - 2D稳态: (x, y, u)

        Args:
            data: MATLAB 数据字典
            x_ref: x 坐标
            mat_file: 文件路径

        Returns:
            处理后的参考解字典
        """
        pass

    # ==================== 钩子方法（可选覆盖） ====================

    def _solver_specific_init(self) -> None:
        """求解器特定初始化 - 钩子方法

        子类可以覆盖此方法进行特定初始化，如：
        - TimePDE: 初始化 loss_history 和动画数据
        - GeneralPDE: 初始化收敛历史
        - FixedPoint: 初始化迭代历史
        """
        pass

    def _on_network_trained(
        self,
        iter: int,
        dt: float,
        U_current: np.ndarray,
        final_loss: float,
        *,
        accepted: bool = True,
    ) -> None:
        """神经网络训练完成后的钩子 - 子类可覆盖

        默认实现为空。TimePDE 等求解器可用来记录 loss_history、触发可视化等。

        Args:
            iter: 当前迭代次数
            dt: 当前时间步长
            U_current: 当前解（训练输入）
            final_loss: 训练后计算得到的物理损失
            accepted: 是否接受本次训练后的模型更新
        """
        # 默认不执行任何操作，由子类根据需要覆盖
        _ = (iter, dt, U_current, final_loss, accepted)

    def _accept_trained_model_update(
        self,
        iter: int,
        dt: float,
        U_current: np.ndarray,
        final_loss: float,
    ) -> bool:
        """Determine whether the newly trained model should replace the previous one."""
        _ = (iter, dt, U_current, final_loss)
        return True

    def _postprocess_results(self, **kwargs: Any) -> None:
        """后处理和可视化 - 钩子方法

        子类可以覆盖此方法进行特定后处理

        Args:
            **kwargs: 子类特定的参数
        """
        # 钩子方法，参数由子类使用
        _ = kwargs

    def _get_initial_guess(self) -> np.ndarray:
        """获取初始猜测值 - 钩子方法

        默认实现：零初值

        Returns:
            初始猜测值 (n_points, n_eqs)
        """
        n_points = self.data_train['x'].shape[0]
        n_eqs = self.config.n_eqs
        return np.zeros((n_points, n_eqs))

    def _initialize_from_config(self) -> np.ndarray:
        """根据配置生成初始猜测值 - 通用方法

        支持三种初始化方法：
        1. 'zero': 零初值
        2. 'spotter': 使用Spotter网络预训练
        3. 'given': 从Initial_conditions表达式计算

        Returns:
            初始猜测值 (n_points, n_eqs)
        """
        # 获取初始化方法配置
        initial_condition = getattr(self.config, 'initial_condition', {})
        if isinstance(initial_condition, dict):
            init_method = initial_condition.get('method', 'zero')
        else:
            init_method = 'zero'

        print(f"Initializing solution using method: '{init_method}'")

        if init_method == 'zero':
            return self._get_initial_guess()
        elif init_method == 'spotter':
            return self._spotter_initialization()
        elif init_method == 'external_spotter':
            return self._external_spotter_initialization()
        elif init_method == 'bc_pinn':
            return self._bc_pinn_initialization()
        elif init_method == 'jaxpi_transfer':
            return self._jaxpi_transfer_initialization()
        elif init_method == 'transfer':
            return self._transfer_initialization()
        elif init_method == 'given':
            return self._config_based_initialization()
        else:
            raise ValueError(f"Unknown initial_condition method: {init_method}")

    def _spotter_initialization(self) -> np.ndarray:
        """使用Spotter网络预训练初始化 - 通用方法

        通用流程：
        1. 创建SpotterNet（使用与主网络相同的架构）
        2. 训练Spotter（最小化物理损失）
        3. 转移权重到主网络
        4. 提取预测值作为初值

        Returns:
            从Spotter网络预测的初始解 (n_points, n_eqs)

        Note:
            子类需要确保有SpotterNet类可用，或覆盖此方法
        """
        print("  Using Spotter network for initialization...")

        # 获取Spotter配置
        spotter_config = getattr(self.config, 'initial_condition', {}).get('spotter_config', {})
        n_epochs = spotter_config.get('epochs', 1000)
        lr_init = spotter_config.get('learning_rate', 1e-2)
        n_pde_points = spotter_config.get('n_pde_points', 10000)
        print_interval = spotter_config.get('print_interval', 200)

        # 尝试导入SpotterNet
        try:
            # 子类特定的SpotterNet导入
            SpotterNet = self._get_spotter_net_class()
        except (NotImplementedError, ImportError) as e:
            print(f"  Warning: SpotterNet not available ({e})")
            print("  Falling back to zero initialization")
            return self._get_initial_guess()

        # 创建并训练Spotter
        spotter_model = SpotterNet(self.config, use_periodic=False).to(self.config.device)

        print(f"  Training SpotterNet with {n_epochs} epochs...")
        print(f"  PDE sampling points: {n_pde_points}")
        print(f"  Initial learning rate: {lr_init}")

        spotter_model.train_spotter(
            data=self.data_train,
            n_epochs=n_epochs,
            lr_init=lr_init,
            n_pde_points=n_pde_points,
            print_every=print_interval
        )

        # 转移权重到主网络
        print("\n  Transferring network weights from SpotterNet to main network...")
        self.model.load_state_dict(spotter_model.state_dict())
        print("  ✓ Network weights transferred successfully")

        # 提取初始解
        U_init = self._predict_current_train_field()
        u_norm = np.linalg.norm(U_init)
        u_max = np.max(np.abs(U_init))
        print(f"  Initial solution: ||U|| = {u_norm:.6e}, max|U| = {u_max:.6e}")
        return U_init

    def _bc_pinn_initialization(self) -> np.ndarray:
        """Initialize with a sequential bc-PINN style spotter."""
        print("  Using bc-PINN style sequential Spotter initialization...")

        try:
            initializer_cls = self._get_bc_pinn_initializer_class()
        except (NotImplementedError, ImportError) as e:
            print(f"  Warning: bc-PINN initializer not available ({e})")
            print("  Falling back to standard spotter initialization")
            return self._spotter_initialization()

        initializer = initializer_cls(self)
        U_init = initializer.run()
        self._skip_initial_network_training = True

        u_norm = np.linalg.norm(U_init)
        u_max = np.max(np.abs(U_init))
        print(f"  bc-PINN initial solution: ||U|| = {u_norm:.6e}, max|U| = {u_max:.6e}")
        return U_init

    def _transfer_initialization(self) -> np.ndarray:
        """Load a warm start from another case and reuse both field and network weights."""
        print("  Using transfer initialization...")

        transfer_config = self._get_transfer_initialization_config()
        self._skip_initial_spotter_training = False
        self._transferred_network_weights_loaded = False
        self._transferred_coeffs_init = None
        results_dir = self._resolve_transfer_results_dir(transfer_config)
        load_network_weights = bool(transfer_config.get("load_network_weights", True))
        load_field = bool(transfer_config.get("load_field", True))
        fallback_to_network_prediction = bool(
            transfer_config.get("fallback_to_network_prediction", True)
        )
        field_blend = float(transfer_config.get("field_blend", 1.0))

        print(f"  Transfer source: {results_dir}")

        if load_network_weights:
            checkpoint_path = self._resolve_transfer_checkpoint_path(transfer_config, results_dir)
            self._load_transfer_network_weights(checkpoint_path)
            self._transferred_network_weights_loaded = True

        if bool(transfer_config.get("skip_initial_training", False)):
            if self._transferred_network_weights_loaded:
                self._skip_initial_spotter_training = True
                print("  Reusing transferred Spotter weights without initial retraining")
            else:
                print("  Warning: skip_initial_training requested without transferred weights; ignoring")

        U_init = None
        if load_field:
            solution_path = self._resolve_transfer_solution_path(transfer_config, results_dir)
            U_init = self._load_transfer_solution_field(solution_path, transfer_config)

        if U_init is None and load_network_weights and fallback_to_network_prediction:
            print("  Falling back to transferred network prediction on current train points")
            U_init = self._predict_current_train_field()

        if U_init is None:
            print("  Warning: transfer initialization unavailable, falling back to zero initialization")
            return self._get_initial_guess()

        if not np.isclose(field_blend, 1.0):
            U_zero = self._get_initial_guess()
            U_init = field_blend * U_init + (1.0 - field_blend) * U_zero
            print(f"  Blended transferred field with zero initialization (field_blend={field_blend:.3f})")

        u_norm = np.linalg.norm(U_init)
        u_max = np.max(np.abs(U_init))
        print(f"  Transferred initial solution: ||U|| = {u_norm:.6e}, max|U| = {u_max:.6e}")
        return U_init

    def _jaxpi_transfer_initialization(self) -> np.ndarray:
        """Initialize from a JAX-PI exported field and optionally distill it into the Torch spotter."""
        print("  Using JAX-PI transfer initialization...")

        jaxpi_config = self._get_jaxpi_transfer_config()
        transfer_config = {
            "results_dir": jaxpi_config["results_dir"],
            "source_case": jaxpi_config.get("source_case"),
            "load_network_weights": False,
            "load_field": True,
            "fallback_to_network_prediction": False,
            "solution_file": jaxpi_config["solution_file"],
            "solution_key": jaxpi_config["solution_key"],
            "x_key": jaxpi_config["x_key"],
            "interpolation": jaxpi_config["interpolation"],
            "fallback_interpolation": jaxpi_config["fallback_interpolation"],
            "field_blend": jaxpi_config["field_blend"],
        }

        results_dir = self._resolve_transfer_results_dir(transfer_config)
        solution_path = self._resolve_transfer_solution_path(transfer_config, results_dir)
        U_init = self._load_transfer_solution_field(solution_path, transfer_config)

        teacher_fit_epochs = int(jaxpi_config.get("teacher_fit_epochs", 0))
        if teacher_fit_epochs > 0:
            self._distill_external_field_to_model(U_init, jaxpi_config)
            if bool(jaxpi_config.get("skip_initial_training", True)):
                self._skip_initial_spotter_training = True
                self._skip_initial_network_training = True
                print("  Reusing Torch Spotter weights distilled from JAX-PI field")

        u_norm = np.linalg.norm(U_init)
        u_max = np.max(np.abs(U_init))
        print(f"  JAX-PI initial solution: ||U|| = {u_norm:.6e}, max|U| = {u_max:.6e}")
        return U_init

    def _get_transfer_initialization_config(self) -> Dict[str, Any]:
        initial_condition = getattr(self.config, "initial_condition", {})
        transfer_config = {}
        if isinstance(initial_condition, dict):
            raw_transfer = initial_condition.get("transfer", {})
            if isinstance(raw_transfer, dict):
                transfer_config.update(raw_transfer)

            aliases = {
                "source_case": ("source_case", "transfer_from"),
                "results_dir": ("results_dir", "source_results_dir"),
                "load_network_weights": ("load_network_weights", "transfer_weights"),
                "load_field": ("load_field", "transfer_field"),
                "checkpoint": ("checkpoint", "checkpoint_name"),
                "solution_file": ("solution_file", "solution_filename"),
                "solution_key": ("solution_key", "transfer_solution_key"),
                "x_key": ("x_key", "transfer_x_key"),
                "interpolation": ("interpolation", "transfer_interpolation"),
                "fallback_interpolation": (
                    "fallback_interpolation",
                    "transfer_fallback_interpolation",
                ),
                "field_blend": ("field_blend", "transfer_field_blend"),
            }
            for target_key, source_keys in aliases.items():
                for source_key in source_keys:
                    if source_key in initial_condition and target_key not in transfer_config:
                        transfer_config[target_key] = initial_condition[source_key]
                        break
        return transfer_config

    def _get_jaxpi_transfer_config(self) -> Dict[str, Any]:
        initial_condition = getattr(self.config, "initial_condition", {})
        if not isinstance(initial_condition, dict):
            raise ValueError("initial_condition must be a dict for method='jaxpi_transfer'")

        raw = initial_condition.get("jaxpi_transfer", {})
        if not isinstance(raw, dict):
            raw = {}

        config = {
            "results_dir": raw.get("results_dir"),
            "source_case": raw.get("source_case"),
            "solution_file": raw.get("solution_file", "jaxpi_transfer.npz"),
            "solution_key": raw.get("solution_key", "U_train"),
            "x_key": raw.get("x_key", "x_train"),
            "interpolation": raw.get("interpolation", "linear"),
            "fallback_interpolation": raw.get("fallback_interpolation", "nearest"),
            "field_blend": float(raw.get("field_blend", 1.0)),
            "teacher_fit_epochs": int(raw.get("teacher_fit_epochs", 2000)),
            "teacher_fit_learning_rate": float(raw.get("teacher_fit_learning_rate", 1e-3)),
            "teacher_fit_print_interval": int(raw.get("teacher_fit_print_interval", 200)),
            "teacher_fit_loss_tol": float(raw.get("teacher_fit_loss_tol", 1e-8)),
            "skip_initial_training": bool(raw.get("skip_initial_training", True)),
        }

        if not config["results_dir"] and not config["source_case"]:
            raise ValueError(
                "initial_condition.method='jaxpi_transfer' requires "
                "'initial_condition.jaxpi_transfer.results_dir' or 'source_case'"
            )
        return config

    def _distill_external_field_to_model(
        self,
        target_field: np.ndarray,
        jaxpi_config: Dict[str, Any],
    ) -> None:
        torch = _import_torch()
        epochs = int(jaxpi_config.get("teacher_fit_epochs", 0))
        if epochs <= 0:
            return

        lr = float(jaxpi_config.get("teacher_fit_learning_rate", 1e-3))
        print_interval = max(1, int(jaxpi_config.get("teacher_fit_print_interval", 200)))
        loss_tol = float(jaxpi_config.get("teacher_fit_loss_tol", 1e-8))

        print(
            "  Distilling JAX-PI field into Torch Spotter "
            f"(epochs={epochs}, lr={lr:.2e}, tol={loss_tol:.2e})"
        )

        parameter = next(self.model.parameters(), None)
        model_dtype = parameter.dtype if parameter is not None else torch.float64
        x_tensor = torch.tensor(self.data_train["x"], dtype=model_dtype, device=self.config.device)
        target_tensor = torch.tensor(target_field, dtype=model_dtype, device=self.config.device)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        best_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(x_tensor)
            pred = output[-1] if isinstance(output, tuple) else output
            loss = criterion(pred, target_tensor)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            if epoch % print_interval == 0 or epoch == epochs - 1:
                print(f"    Teacher fit epoch {epoch:6d} | mse={loss_value:.6e}")

            if loss_value <= loss_tol:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        print(f"  ✓ JAX-PI teacher distillation completed (best_mse={best_loss:.6e})")

    def _resolve_transfer_results_dir(self, transfer_config: Dict[str, Any]) -> str:
        base_dir = getattr(self.config, "case_dir", None) or self.case_dir or os.getcwd()

        if transfer_config.get("results_dir"):
            raw_path = transfer_config["results_dir"]
        else:
            raw_path = transfer_config.get("source_case")

        if not raw_path:
            raise ValueError(
                "initial_condition.method='transfer' requires 'transfer.source_case' "
                "or 'transfer.results_dir'"
            )

        resolved = raw_path if os.path.isabs(raw_path) else os.path.abspath(os.path.join(base_dir, raw_path))
        if os.path.isdir(resolved):
            try:
                resolve_case_config_path(resolved)
                resolved = os.path.join(resolved, "results")
            except (FileNotFoundError, FileExistsError):
                pass
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"Transfer results directory not found: {resolved}")
        return resolved

    def _resolve_transfer_checkpoint_path(
        self,
        transfer_config: Dict[str, Any],
        results_dir: str,
    ) -> str:
        checkpoint_name = transfer_config.get("checkpoint", "spotter_model.pth")
        checkpoint_path = (
            checkpoint_name
            if os.path.isabs(checkpoint_name)
            else os.path.join(results_dir, checkpoint_name)
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Transfer checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    def _resolve_transfer_solution_path(
        self,
        transfer_config: Dict[str, Any],
        results_dir: str,
    ) -> str:
        solution_name = transfer_config.get("solution_file", "solution_data.npz")
        solution_path = (
            solution_name
            if os.path.isabs(solution_name)
            else os.path.join(results_dir, solution_name)
        )
        if not os.path.isfile(solution_path):
            raise FileNotFoundError(f"Transfer solution file not found: {solution_path}")
        return solution_path

    def _load_transfer_network_weights(self, checkpoint_path: str) -> None:
        torch = _import_torch()
        print(f"  Loading transferred network weights: {checkpoint_path}")
        try:
            state = torch.load(
                checkpoint_path,
                map_location=self.config.device,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(checkpoint_path, map_location=self.config.device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
        self.model.load_state_dict(state)
        print("  ✓ Transferred network weights loaded")

    def _load_transfer_solution_field(
        self,
        solution_path: str,
        transfer_config: Dict[str, Any],
    ) -> np.ndarray:
        from scipy.interpolate import griddata

        solution_key = transfer_config.get("solution_key", "sniper_train")
        x_key = transfer_config.get("x_key", "x_train")
        interpolation = transfer_config.get("interpolation", "linear")
        fallback_interpolation = transfer_config.get("fallback_interpolation", "nearest")

        print(f"  Loading transferred field: {solution_path} [{solution_key}]")
        payload = np.load(solution_path)
        if bool(transfer_config.get("load_coefficients", False)):
            coeffs = payload.get("coeffs")
            if coeffs is None:
                print(f"  Warning: no 'coeffs' found in {solution_path}; falling back to zero coefficients")
            else:
                self._transferred_coeffs_init = np.asarray(coeffs, dtype=np.float64).copy()
                print(f"  Loaded transferred Snipper coefficients: shape={self._transferred_coeffs_init.shape}")
        if solution_key not in payload:
            raise KeyError(f"Transfer solution key '{solution_key}' not found in {solution_path}")

        U_source = np.asarray(payload[solution_key], dtype=np.float64)
        if U_source.ndim == 1:
            U_source = U_source.reshape(-1, 1)

        x_target = np.asarray(self.data_train["x"], dtype=np.float64)
        if x_key in payload:
            x_source = np.asarray(payload[x_key], dtype=np.float64)
        else:
            x_source = None

        if x_source is not None and x_source.shape == x_target.shape:
            if np.allclose(x_source, x_target, atol=1e-12, rtol=1e-12):
                print("  Transfer points match current train points exactly")
                return U_source.copy()

        if x_source is None:
            if U_source.shape[0] != x_target.shape[0]:
                raise ValueError(
                    f"Transfer field has {U_source.shape[0]} points but current case uses "
                    f"{x_target.shape[0]} points, and no '{x_key}' was provided for interpolation"
                )
            print("  Transfer field has no coordinates; reusing values by point index")
            return U_source.copy()

        print("  Transfer points differ from current train points, interpolating field")
        U_interp = np.zeros((x_target.shape[0], U_source.shape[1]), dtype=np.float64)
        for eq_idx in range(U_source.shape[1]):
            interpolated = griddata(
                x_source,
                U_source[:, eq_idx],
                x_target,
                method=interpolation,
                fill_value=np.nan,
            )
            values = interpolated.reshape(-1)
            nan_mask = np.isnan(values)
            if np.any(nan_mask):
                values[nan_mask] = griddata(
                    x_source,
                    U_source[:, eq_idx],
                    x_target[nan_mask],
                    method=fallback_interpolation,
                ).reshape(-1)
            U_interp[:, eq_idx] = values
        return U_interp

    def _predict_current_train_field(self) -> np.ndarray:
        return self._predict_field_on_points(self.data_train["x"])

    def _predict_field_on_points(self, points: np.ndarray) -> np.ndarray:
        feature_source = self._get_spotter_feature_source()
        if hasattr(feature_source, "predict_output"):
            return np.asarray(feature_source.predict_output(points), dtype=np.float64)
        torch = _import_torch()
        parameter = next(self.model.parameters(), None)
        model_dtype = parameter.dtype if parameter is not None else torch.float64
        x_tensor = torch.tensor(
            points,
            dtype=model_dtype,
            device=self.config.device,
        )
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_tensor)
        U_pred = output[-1] if isinstance(output, tuple) else output
        return self._to_numpy_cpu(U_pred)

    def _external_spotter_initialization(self) -> np.ndarray:
        """Initialize from the active spotter prediction without Torch retraining."""
        print("  Using external spotter prediction for initialization...")
        U_init = self._predict_current_train_field()
        u_norm = np.linalg.norm(U_init)
        u_max = np.max(np.abs(U_init)) if U_init.size else 0.0
        print(f"  External Spotter field loaded: ||U|| = {u_norm:.6e}, max|U| = {u_max:.6e}")
        return np.asarray(U_init, dtype=np.float64)

    def _config_based_initialization(self) -> np.ndarray:
        """根据Initial_conditions表达式计算初值 - 通用方法

        从config.Initial_conditions读取表达式，如：
        ```yaml
        Initial_conditions:
          - var: u
            value: "sin(2*pi*x[:, 0])"
          - var: v
            value: "cos(2*pi*x[:, 1])"
        ```

        Returns:
            从表达式计算的初始解 (n_points, n_eqs)
        """
        if not hasattr(self.config, 'Initial_conditions'):
            raise ValueError("No Initial_conditions found in config for method='given'")

        print("  Using given initial conditions from config...")

        n_points = self.data_train['x'].shape[0]
        n_eqs = self.config.n_eqs
        U_init = np.zeros((n_points, n_eqs))

        # 获取空间坐标
        x_coords = self.data_train['x']

        # 处理每个初始条件
        for ic in self.config.Initial_conditions:
            var_name = ic.get('var')
            value_expr = ic.get('value')

            # 找到变量索引
            if not hasattr(self.config, 'vars_list'):
                # 如果没有vars_list，假设变量按顺序排列
                var_idx = int(var_name.replace('u', '').replace('v', '').replace('w', '') or '0')
            else:
                var_idx = self.config.vars_list.index(var_name)

            # 构建命名空间用于eval
            namespace = {
                'np': np,
                'pi': np.pi,
                'cos': np.cos,
                'sin': np.sin,
                'exp': np.exp,
                'sqrt': np.sqrt,
                'tanh': np.tanh,
                'x': x_coords  # 提供完整的x数组
            }

            # 如果有空间变量名称列表，添加每个维度
            if hasattr(self.config, 'spatial_vars'):
                for i, var in enumerate(self.config.spatial_vars):
                    namespace[var] = x_coords[:, i]

            # 计算表达式
            try:
                U_init[:, var_idx] = eval(value_expr, namespace)
                print(f"  Initialized {var_name} with: {value_expr}")

                u_norm = np.linalg.norm(U_init[:, var_idx])
                u_max = np.max(np.abs(U_init[:, var_idx]))
                print(f"    ||{var_name}|| = {u_norm:.6e}, max|{var_name}| = {u_max:.6e}")
            except Exception as e:
                print(f"  Warning: Failed to evaluate expression for {var_name}: {e}")
                print(f"  Using zero initialization for {var_name}")

        return U_init

    def _get_spotter_net_class(self):
        """获取SpotterNet类 - 钩子方法

        子类应该覆盖此方法以提供自己的SpotterNet实现

        Returns:
            SpotterNet类（未实例化）

        Raises:
            NotImplementedError: 如果子类没有提供SpotterNet
        """
        raise NotImplementedError("Subclass must provide SpotterNet class or override _spotter_initialization()")

    def _get_bc_pinn_initializer_class(self):
        """Hook for subclasses to provide a bc-PINN initializer implementation."""
        raise NotImplementedError("Subclass must provide bc-PINN initializer or override _bc_pinn_initialization()")

    def _adapt_pseudo_timestep(self, dt: float, residual: float) -> float:
        """自适应伪时间步调整 - 钩子方法

        默认实现：不调整（返回原始dt）
        GeneralPDESolver 应该覆盖此方法实现自适应策略

        Args:
            dt: 当前时间步长
            residual: 当前残差

        Returns:
            调整后的时间步长
        """
        return dt

    # ==================== 时间步控制方法（新增 - 从Fitter移到Solver层） ====================

    def validate_operator_configuration(self) -> Dict[str, bool]:
        """验证算子配置 - 通用方法

        检查fitter的time_scheme和算子配置是否有效。
        所有solver都可以使用此方法验证配置。

        Returns:
            验证结果字典，包含验证状态和可能的错误信息
        """
        if not hasattr(self.fitter, 'time_scheme') or self.fitter.time_scheme is None:
            return {
                "valid": False,
                "error": "No time scheme initialized"
            }

        return self.fitter.time_scheme.validate_operators()

    def get_solver_time_info(self) -> Dict[str, Any]:
        """获取求解器时间信息 - 基础接口（子类覆盖）

        返回求解器的时间步相关信息。子类应该覆盖此方法
        提供特定的时间信息。

        Returns:
            时间信息字典，包含solver类型和时间步配置等
        """
        return {
            "solver_type": self.__class__.__name__,
            "has_time_scheme": hasattr(self.fitter, 'time_scheme') and self.fitter.time_scheme is not None
        }

    def _get_network_prediction(self) -> np.ndarray:
        """从训练后的网络提取预测值 - 通用方法

        用于混合网络方法（Spotter + Sniper）：
        - Spotter（神经网络）提供初始猜测
        - Sniper（多项式）基于这个猜测进行优化

        Returns:
            网络预测的解 (n_points, n_eqs)
        """
        return self._predict_field_on_points(self.data_train["x"])

    def _create_spotter_session(self):
        return create_spotter_session(
            config=self.config,
            model=self.model,
            data_train=getattr(self, "data_train", None),
            data_test=getattr(self, "data_test", None),
            reference_solution=getattr(self, "reference_solution", None),
        )

    def _create_spotter_feature_provider(self):
        session = getattr(self, "spotter_session", None)
        if session is None:
            return None
        return getattr(session, "provider", None)

    def _create_spotter_orchestrator(self):
        session = getattr(self, "spotter_session", None)
        if session is None:
            return None
        return getattr(session, "orchestrator", None)

    def _sync_spotter_session(self, *, model_updated: bool = False) -> None:
        session = getattr(self, "spotter_session", None)
        if session is None:
            return
        session.sync(model_updated=model_updated)
        self.spotter_feature_provider = getattr(session, "provider", None)
        self.spotter_orchestrator = getattr(session, "orchestrator", None)
        self._save_spotter_selection_report()

    def _maybe_reselect_spotter_provider(self) -> None:
        self._sync_spotter_session(model_updated=False)

    def _save_spotter_selection_report(self) -> None:
        session = getattr(self, "spotter_session", None)
        if session is None:
            return
        results_dir = getattr(self.config, "results_dir", None)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            session.save_report(results_dir)

    def _get_spotter_feature_source(self):
        session = getattr(self, "spotter_session", None)
        if session is None:
            return self.model
        return session.feature_source()

    def _needs_live_torch_model(self) -> bool:
        if getattr(self.config, "spotter_feature_backend", "torch") == "torch":
            return True

        raw_spotter = getattr(self.config, "spotter", {}) or {}
        if raw_spotter:
            for item in raw_spotter.get("candidates", []) or []:
                if not isinstance(item, dict) or not bool(item.get("enabled", True)):
                    continue
                backend = str(item.get("backend", "torch") or "torch").lower()
                if backend == "torch" or bool(item.get("allow_torch_training", False)):
                    return True

        legacy_cfg = getattr(self.config, "spotter_feature", {}) or {}
        if bool(legacy_cfg.get("allow_torch_training", False)):
            return True

        initial_condition = getattr(self.config, "initial_condition", {}) or {}
        init_method = ""
        if isinstance(initial_condition, dict):
            init_method = str(initial_condition.get("method", "") or "").lower()
        if init_method in {"spotter", "bc_pinn", "transfer", "jaxpi_transfer"}:
            return True

        return False

    def _build_external_only_spotter_model(self):
        return NullSpotterModel()

    def _uses_external_spotter_feature_backend(self) -> bool:
        session = getattr(self, "spotter_session", None)
        if session is None:
            return False
        return session.uses_external_backend()

    def _allow_torch_training_with_external_features(self) -> bool:
        session = getattr(self, "spotter_session", None)
        if session is not None:
            return session.allow_local_torch_training()
        cfg = getattr(self.config, "spotter_feature", {}) or {}
        return bool(cfg.get("allow_torch_training", False))

    def _can_train_external_spotter_backend(self) -> bool:
        session = getattr(self, "spotter_session", None)
        if session is None:
            return False
        supports = getattr(session, "can_train_externally", None)
        if callable(supports):
            return bool(supports())
        return bool(supports)

    def _train_external_spotter_step(
        self,
        *,
        iter: int,
        dt: float,
        U_current: np.ndarray,
    ):
        session = getattr(self, "spotter_session", None)
        if session is None:
            raise RuntimeError("No active spotter session is available for external training.")
        return session.train_external_step(
            problem_type=str(getattr(self.config, "problem_type", "")),
            dt=float(dt),
            U_current=np.asarray(U_current, dtype=np.float64),
            iteration=int(iter),
        )

    def _get_active_spotter_backend_name(self) -> str:
        session = getattr(self, "spotter_session", None)
        if session is not None:
            return session.active_backend_name()
        return str(getattr(self.config, "spotter_feature_backend", "torch") or "torch").lower()

    def __del__(self):  # pragma: no cover - defensive cleanup
        session = getattr(self, "spotter_session", None)
        if session is None:
            return
        try:
            session.close()
        except Exception:
            pass
