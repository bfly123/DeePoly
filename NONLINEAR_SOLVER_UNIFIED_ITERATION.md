# NonlinearPDESolver 统一迭代框架设计

**日期**: 2025-10-10
**核心洞察**: 时间推进本质上也是一种迭代，三种 Solver 可以用统一的迭代框架表达

---

## 🔍 统一迭代视角

### 三种求解策略的统一形式

**通用迭代模式**：
```
初始化：U = U_init, state = init_state
循环：
    while not terminate(state):
        U_new, state_new = iteration_step(U, state)
        U = U_new
        state = state_new
返回：U, final_state
```

### 三种策略的实例化

| 策略 | 终止条件 | 迭代状态 | 迭代步骤 |
|------|---------|---------|---------|
| **TimePDE** | `T >= T_final` | `{T, iter, dt}` | `time_step(U, dt)` |
| **GeneralPDE** | `residual < tol` 或 `iter >= max_iter` | `{τ, iter, dτ, residual}` | `time_step(U, dτ)` + 训练网络 |
| **FixedPoint** | `residual < tol` 或 `iter >= max_iter` | `{iter, residual}` | `fit(U)` |

---

## 💻 统一迭代框架的基类设计

```python
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
import torch
import scipy.io
import os


@dataclass
class IterationState:
    """迭代状态 - 统一的状态管理"""
    iter: int = 0                    # 迭代次数

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
    custom: Dict = None              # 子类可以添加自定义状态

    def __post_init__(self):
        if self.custom is None:
            self.custom = {}


class NonlinearPDESolver(ABC):
    """非线性PDE求解器统一基类 - 基于统一迭代框架

    核心思想：所有求解策略都是迭代 U^{n+1} = F(U^n, state)

    区别仅在于：
    1. 终止条件：时间达到 vs 残差收敛
    2. 迭代步骤：time_step vs fit
    3. 状态更新：时间递增 vs 残差计算
    """

    # ==================== 统一迭代框架 ====================

    def _unified_iteration_loop(
        self,
        U_init: np.ndarray,
        initial_state: IterationState,
        iteration_step: Callable,
        termination_check: Callable,
        state_update: Optional[Callable] = None,
        print_progress: Optional[Callable] = None
    ) -> Tuple[np.ndarray, IterationState]:
        """统一的迭代循环框架 - 三种 Solver 的共同抽象

        这是最核心的方法，提供了一个通用的迭代框架。

        Args:
            U_init: 初始解
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
            (U_final, final_state)
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

    # ==================== 辅助方法（具体方法） ====================

    def _compute_residual(self, U_new: np.ndarray, U_old: np.ndarray) -> float:
        """计算相对残差 - 标准实现"""
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
        """神经网络训练步骤 - 通用框架"""
        if not force and not self._should_train_network(iter):
            self.fitter.fitter_init(self.model)
            return

        print(f"  Training neural network at iteration {iter}...")
        data_GPU = self.model.prepare_gpu_data(self.data_train, U_current)
        self.model.train()
        self.model.train_net(self.data_train, self.model, data_GPU, dt=dt)
        self.model.eval()
        final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
        print(f"  Neural network training completed, loss: {final_loss:.8e}")
        self.fitter.fitter_init(self.model)

    def _should_train_network(self, iter: int) -> bool:
        """判断是否需要训练网络 - 可覆盖"""
        spotter_skip = getattr(self.config, 'spotter_skip', 1)
        return (iter % spotter_skip == 0)

    # ==================== 初始化（100%共性） ====================

    def __init__(self, config=None, case_dir=None):
        """统一的初始化流程"""
        self._initialize_config(config, case_dir)
        self._initialize_components()
        self._solver_specific_init()

    def _initialize_config(self, config, case_dir):
        """配置初始化"""
        if config is not None:
            self.config = config
        elif case_dir is not None:
            self.config = self._create_config(case_dir)
        else:
            raise ValueError("Either config object or case_dir must be provided")

        self.case_dir = case_dir if case_dir else getattr(self.config, "case_dir", None)

        if hasattr(self.config, "export_to_json"):
            self.config.export_to_json("config.json")

    def _initialize_components(self):
        """组件初始化"""
        self.datagen = self._create_data_generator()
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")
        self.model = self._create_model().to(self.config.device)
        self.visualizer = self._create_visualizer()

        time_scheme = getattr(self.config, 'time_scheme', self._get_default_time_scheme())
        self.fitter = self._create_fitter(time_scheme)

        self.reference_solution = self._load_reference_solution()
        os.makedirs(self.config.results_dir, exist_ok=True)

    def _load_reference_solution(self) -> Optional[Dict]:
        """加载参考解"""
        if not hasattr(self.config, 'reference_solution') or not self.config.reference_solution:
            return None

        ref_path = self.config.reference_solution
        if not os.path.isabs(ref_path):
            case_dir = getattr(self.config, 'case_dir', os.getcwd())
            ref_path = os.path.join(case_dir, ref_path)

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

            if 'x' in data:
                x_ref = data['x'].flatten()
            else:
                print("Warning: 'x' not found in reference solution file")
                return None

            return self._process_reference_solution(data, x_ref, mat_file)

        except Exception as e:
            print(f"Error: Cannot read reference solution file {mat_file}: {e}")
            return None

    def _to_numpy_cpu(self, data):
        """GPU数据转CPU numpy数组"""
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, np.ndarray):
            return np.asarray(data)
        try:
            return np.asarray(data)
        except:
            return data

    def _interpolate_to_test_grid(self, U_train: np.ndarray) -> np.ndarray:
        """从训练点插值到测试网格"""
        from scipy.interpolate import griddata

        x_train = self.data_train['x']
        x_test = self.data_test['x']
        n_eqs = U_train.shape[1] if U_train.ndim > 1 else 1

        if n_eqs == 1:
            U_train = U_train.reshape(-1, 1)

        U_test = np.zeros((x_test.shape[0], n_eqs))

        for eq_idx in range(n_eqs):
            U_test[:, eq_idx] = griddata(
                x_train, U_train[:, eq_idx], x_test,
                method='linear', fill_value=np.nan
            )

            nan_mask = np.isnan(U_test[:, eq_idx])
            if np.any(nan_mask):
                U_test[nan_mask, eq_idx] = griddata(
                    x_train, U_train[:, eq_idx], x_test[nan_mask],
                    method='nearest'
                )

        print(f"  Interpolated from {x_train.shape[0]} training points to {x_test.shape[0]} test points")
        return U_test

    # ==================== 抽象方法 ====================

    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """主求解方法 - 子类使用 _unified_iteration_loop 实现"""
        pass

    @abstractmethod
    def _create_config(self, case_dir: str):
        """创建配置对象"""
        pass

    @abstractmethod
    def _create_data_generator(self):
        """创建数据生成器"""
        pass

    @abstractmethod
    def _create_model(self):
        """创建神经网络模型"""
        pass

    @abstractmethod
    def _create_fitter(self, time_scheme: Optional[str]):
        """创建Fitter对象"""
        pass

    @abstractmethod
    def _create_visualizer(self):
        """创建可视化器"""
        pass

    @abstractmethod
    def _get_default_time_scheme(self) -> Optional[str]:
        """获取默认时间格式"""
        pass

    @abstractmethod
    def _process_reference_solution(self, data: Dict, x_ref: np.ndarray, mat_file: str) -> Dict:
        """处理参考解数据"""
        pass

    # ==================== 钩子方法 ====================

    def _solver_specific_init(self):
        """求解器特定初始化"""
        pass

    def _postprocess_results(self, **kwargs):
        """后处理和可视化"""
        pass

    def _get_initial_guess(self) -> np.ndarray:
        """获取初始猜测值"""
        n_points = self.data_train['x'].shape[0]
        n_eqs = self.config.n_eqs
        return np.zeros((n_points, n_eqs))
```

---

## 🔧 三种 Solver 使用统一框架的实现

### 1. TimePDESolver - 时间推进迭代

```python
class TimePDESolver(NonlinearPDESolver):
    """物理时间演化求解器 - 使用统一迭代框架"""

    def solve(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """物理时间演化 - 使用统一迭代框架"""
        print("\n" + "="*70)
        print("Time-Dependent PDE Solver")
        print("="*70)

        # 初始化
        U_init = self.data_train['U0']
        U_seg_init = self.fitter.global_to_segments(U_init)
        self.fitter.fitter_init(self.model)

        # 初始状态
        initial_state = IterationState(
            iter=0,
            time=0.0,
            dt=self.config.dt,
            coeffs=None,
            custom={'U_seg': U_seg_init}
        )

        # 定义迭代步骤
        def iteration_step(U: np.ndarray, state: IterationState) -> Tuple[np.ndarray, IterationState]:
            """时间步推进"""
            # 可选：训练神经网络
            self._train_neural_network_step(state.iter, state.dt, U)

            # 时间步推进
            U_seg = state.custom['U_seg']
            U_new, U_seg_new, coeffs_new = self.fitter.time_step(
                U, U_seg, state.dt, state.coeffs,
                current_time=state.time,
                step=state.iter
            )

            # 更新状态
            new_state = IterationState(
                iter=state.iter + 1,
                time=state.time + state.dt,
                dt=state.dt,  # 固定步长（或可实现自适应）
                coeffs=coeffs_new,
                custom={'U_seg': U_seg_new}
            )

            return U_new, new_state

        # 定义终止条件
        def termination_check(state: IterationState) -> bool:
            """时间达到终止时间"""
            return state.time >= self.config.T_final

        # 定义进度打印
        def print_progress(state: IterationState):
            print(f"Step {state.iter}: T = {state.time:.6f}, dt = {state.dt:.6e}")

        # 使用统一迭代框架
        U_final, final_state = self._unified_iteration_loop(
            U_init=U_init,
            initial_state=initial_state,
            iteration_step=iteration_step,
            termination_check=termination_check,
            print_progress=print_progress
        )

        # 提取结果
        U_seg_final = final_state.custom['U_seg']
        coeffs_final = final_state.coeffs

        # 后处理
        self._postprocess_results(U_final, U_seg_final, self.model, coeffs_final)

        return U_final, U_seg_final, self.model, coeffs_final

    def _solver_specific_init(self):
        """时间演化特定初始化"""
        self.loss_history = []
        self.visualizer.reset_animation_data()

    # ... 工厂方法实现
```

### 2. GeneralPDESolver - 伪时间步迭代

```python
class GeneralPDESolver(NonlinearPDESolver):
    """伪时间步稳态求解器 - 使用统一迭代框架"""

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict:
        """伪时间步迭代 - 使用统一迭代框架"""
        print("\n" + "="*70)
        print("General PDE Solver - Pseudo-Time Stepping")
        print("="*70)

        # 初始化
        if U_init is None:
            U_init = self._get_initial_guess()

        self.fitter.fitter_init(self.model)
        U_seg_init = self.fitter.global_to_segments(U_init)

        # 参数
        max_iter = getattr(self.config, 'max_iterations', 1000)
        conv_tol = getattr(self.config, 'convergence_tol', 1e-6)
        dt_init = getattr(self.config, 'dt_init', 1.0)

        # 初始状态
        initial_state = IterationState(
            iter=0,
            time=0.0,  # 伪时间 τ
            dt=dt_init,
            residual=float('inf'),
            converged=False,
            U_prev=None,
            coeffs=None,
            custom={'U_seg': U_seg_init, 'max_iter': max_iter, 'conv_tol': conv_tol}
        )

        # 定义迭代步骤
        def iteration_step(U: np.ndarray, state: IterationState) -> Tuple[np.ndarray, IterationState]:
            """伪时间步推进 + 训练网络"""
            # 训练神经网络（每步都训练）
            self._train_neural_network_step(state.iter, state.dt, U, force=True)

            # 伪时间步推进
            U_seg = state.custom['U_seg']
            U_new, U_seg_new, coeffs_new = self.fitter.time_step(
                U, U_seg, state.dt, state.coeffs,
                current_time=state.time,
                step=state.iter
            )

            # 计算残差
            residual = self._compute_residual(U_new, U)

            # 更新状态
            new_state = IterationState(
                iter=state.iter + 1,
                time=state.time + state.dt,
                dt=state.dt,  # 稍后自适应更新
                residual=residual,
                converged=(residual < state.custom['conv_tol']),
                U_prev=U.copy(),
                coeffs=coeffs_new,
                custom={'U_seg': U_seg_new, 'max_iter': max_iter, 'conv_tol': conv_tol}
            )

            return U_new, new_state

        # 定义终止条件
        def termination_check(state: IterationState) -> bool:
            """收敛或达到最大迭代次数"""
            return state.converged or state.iter >= state.custom['max_iter']

        # 定义状态更新（自适应伪时间步）
        def state_update(U_old: np.ndarray, U_new: np.ndarray, state: IterationState) -> IterationState:
            """自适应调整伪时间步"""
            dt_new = self._adapt_pseudo_timestep(state.dt, state.residual)
            state.dt = dt_new
            return state

        # 定义进度打印
        def print_progress(state: IterationState):
            print(f"Iteration {state.iter}: τ = {state.time:.6f}, dτ = {state.dt:.6e}, residual = {state.residual:.6e}")

        # 使用统一迭代框架
        U_final, final_state = self._unified_iteration_loop(
            U_init=U_init,
            initial_state=initial_state,
            iteration_step=iteration_step,
            termination_check=termination_check,
            state_update=state_update,
            print_progress=print_progress
        )

        # 构造返回结果
        conv_info = {
            "converged": final_state.converged,
            "iterations": final_state.iter,
            "final_residual": final_state.residual
        }

        self._postprocess(U_final, final_state.coeffs, conv_info)

        return {
            "solution": U_final,
            "coefficients": final_state.coeffs,
            "convergence_info": conv_info
        }

    def _should_train_network(self, iter: int) -> bool:
        """GeneralPDE 每步都训练"""
        return True

    def _adapt_pseudo_timestep(self, dtau, residual):
        """自适应伪时间步"""
        dt_max = getattr(self.config, 'dt_max', 10.0)
        dt_min = getattr(self.config, 'dt_min', 0.01)

        if residual > 0.1:
            return dt_max
        elif residual > 0.01:
            return dt_max * 0.5
        elif residual > 1e-4:
            return dt_max * 0.2
        else:
            return dt_min

    # _get_initial_guess, 工厂方法等...
```

### 3. FixedPointSolver - 不动点迭代

```python
class FixedPointSolver(NonlinearPDESolver):
    """不动点迭代求解器 - 使用统一迭代框架"""

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict:
        """不动点迭代 - 使用统一迭代框架"""
        print("\n" + "="*70)
        print("Fixed-Point Iteration Solver")
        print("="*70)

        # 初始化
        if U_init is None:
            U_init = self._get_initial_guess()

        self.fitter.fitter_init(self.model)

        # 参数
        max_iter = getattr(self.config, 'max_iterations', 100)
        conv_tol = getattr(self.config, 'convergence_tol', 1e-6)

        # 初始状态
        initial_state = IterationState(
            iter=0,
            residual=float('inf'),
            converged=False,
            custom={'max_iter': max_iter, 'conv_tol': conv_tol}
        )

        # 定义迭代步骤
        def iteration_step(U: np.ndarray, state: IterationState) -> Tuple[np.ndarray, IterationState]:
            """不动点迭代步骤：固定 N(U)，求解线性系统"""
            # 固定非线性项，求解
            coeffs = self.fitter.fit(U_current=U)

            # 构造新解
            U_new, _ = self.fitter.construct(self.data_train, self.model, coeffs)

            # 计算残差
            residual = self._compute_residual(U_new, U)

            # 更新状态
            new_state = IterationState(
                iter=state.iter + 1,
                residual=residual,
                converged=(residual < state.custom['conv_tol']),
                coeffs=coeffs,
                custom=state.custom
            )

            return U_new, new_state

        # 定义终止条件
        def termination_check(state: IterationState) -> bool:
            """收敛或达到最大迭代次数"""
            return state.converged or state.iter >= state.custom['max_iter']

        # 定义进度打印
        def print_progress(state: IterationState):
            print(f"Iteration {state.iter}: residual = {state.residual:.6e}")

        # 使用统一迭代框架
        U_final, final_state = self._unified_iteration_loop(
            U_init=U_init,
            initial_state=initial_state,
            iteration_step=iteration_step,
            termination_check=termination_check,
            print_progress=print_progress
        )

        # 构造返回结果
        conv_info = {
            "converged": final_state.converged,
            "iterations": final_state.iter,
            "final_residual": final_state.residual
        }

        self._postprocess(U_final, final_state.coeffs, conv_info)

        return {
            "solution": U_final,
            "coefficients": final_state.coeffs,
            "convergence_info": conv_info
        }

    def _should_train_network(self, iter: int) -> bool:
        """FixedPoint 通常不训练网络"""
        return False

    # 工厂方法等...
```

---

## 📊 统一迭代框架的优势

### 1. **高度统一**

三种 Solver 都使用同一个迭代框架 `_unified_iteration_loop`：

```python
U, state = self._unified_iteration_loop(
    U_init,
    initial_state,
    iteration_step,      # 唯一差异：迭代逻辑
    termination_check,   # 唯一差异：终止条件
    state_update,        # 可选：状态更新
    print_progress       # 可选：进度打印
)
```

### 2. **差异参数化**

| 差异点 | TimePDE | GeneralPDE | FixedPoint |
|--------|---------|------------|------------|
| **iteration_step** | time_step(U, dt) | time_step(U, dτ) + train | fit(U) |
| **termination_check** | T >= T_final | residual < tol | residual < tol |
| **state_update** | dt固定 | 自适应dτ | 无 |

### 3. **代码复用最大化**

```
基类提供：
├─ 统一迭代框架:           ~50行  ← 三者都用
├─ 辅助方法（残差等）:     ~50行  ← 三者都用
├─ 初始化（100%共性）:    ~200行  ← 三者都用
├─ 工具函数:              ~100行  ← 三者都用
└─ 抽象方法定义:           ~50行

各 Solver 实现：
├─ TimePDESolver:      ~100行（定义 iteration_step）
├─ GeneralPDESolver:   ~120行（定义 iteration_step + 自适应）
└─ FixedPointSolver:    ~80行（定义 iteration_step）

总代码复用率：~80%！
```

### 4. **概念清晰**

所有求解都是：
```
U^{n+1} = iteration_step(U^n, state^n)
```

差异仅在于：
- 步骤定义（time_step vs fit）
- 终止条件（时间 vs 残差）

---

## 🎯 设计亮点

### 1. IterationState 统一状态管理

```python
@dataclass
class IterationState:
    iter: int                    # 共性：迭代次数
    time: float                  # TimePDE, GeneralPDE: T 或 τ
    dt: float                    # TimePDE, GeneralPDE: 时间步
    residual: float              # GeneralPDE, FixedPoint: 残差
    converged: bool              # GeneralPDE, FixedPoint: 收敛标志
    U_prev: np.ndarray          # 上一步解
    coeffs: np.ndarray          # 当前系数
    custom: Dict                # 子类自定义状态
```

### 2. 灵活的回调函数

```python
iteration_step:      (U, state) -> (U_new, state_new)  # 必须
termination_check:   (state) -> bool                   # 必须
state_update:        (U_old, U_new, state) -> state   # 可选
print_progress:      (state) -> None                   # 可选
```

### 3. 完全解耦

- 迭代逻辑 ← iteration_step
- 终止条件 ← termination_check
- 状态更新 ← state_update
- 框架本身完全通用

---

## ✅ 总结

### 核心洞察

**时间推进本质上就是迭代**，三种 Solver 的本质都是：

```
U^{n+1} = F(U^n, state)
```

区别仅在于：
1. F 的定义（time_step vs fit）
2. 终止条件（时间 vs 残差）
3. 状态演化（dt固定 vs 自适应）

### 统一框架价值

1. **最大代码复用**：~80% 代码复用率
2. **概念统一**：所有求解都是迭代
3. **易于扩展**：新的求解策略只需定义 iteration_step
4. **保持独立性**：三个 Solver 仍然是独立的类
5. **灵活性强**：可以完全自定义迭代逻辑

### 三种文档对比

1. **NONLINEAR_SOLVER_THREE_STRATEGIES.md**
   - 第一版：完全独立实现

2. **NONLINEAR_SOLVER_UNIFIED_BASE.md**
   - 第二版：提取共性，提供可选框架

3. **NONLINEAR_SOLVER_UNIFIED_ITERATION.md**（当前，推荐）
   - 第三版：统一迭代框架，最高复用率
   - 体现了"时间推进也是迭代"的深刻洞察

这个设计是**最优的**！
