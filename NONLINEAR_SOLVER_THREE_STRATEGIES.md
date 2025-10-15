# NonlinearPDESolver 架构设计：支持三种求解策略

**日期**: 2025-10-10
**目标**: 设计统一的 NonlinearPDESolver 基类，支持三种不同的非线性PDE求解策略

---

## 📊 三种求解策略的本质差异

### 策略1：TimePDESolver - 物理时间演化

```python
# 主循环特征
T = 0.0
U = U_init
while T < T_final:  # 固定终止时间
    # 可选：训练神经网络
    if should_train(iter):
        model.train_net(data, U_current, dt=dt)
        fitter.fitter_init(model)

    # 核心：时间步推进
    U, U_seg, coeffs = fitter.time_step(U, U_seg, dt, coeffs, ...)

    T += dt
```

**特征**：
- ✅ 使用 `fitter.time_step()`
- ✅ 时间格式：IMEX-RK(2,2,2), IMEX-1st
- ✅ 物理时间 T ∈ [0, T_final]
- ✅ CFL条件自适应步长
- ✅ 每步可选神经网络训练

---

### 策略2：GeneralPDESolver - 伪时间步稳态

```python
# 主循环特征
tau = 0.0
U = spotter_initialization()  # Spotter初始化
dtau = dt_init
for iter in range(max_iterations):
    # 必须：训练神经网络
    model.train_net(data, U_current, dt=dtau)
    fitter.fitter_init(model)

    # 核心：伪时间步推进
    U, U_seg, coeffs = fitter.time_step(U, U_seg, dtau, coeffs, ...)

    # 收敛检查
    residual = ||U_new - U_old|| / ||U_old||
    if residual < tol:
        break  # 收敛终止

    # 自适应伪时间步
    dtau = adapt_pseudo_timestep(dtau, residual)
    tau += dtau
```

**特征**：
- ✅ 使用 `fitter.time_step()`
- ✅ 时间格式：implicit_euler
- ✅ 伪时间 τ ∈ [0, ∞)，收敛终止
- ✅ 基于残差的自适应伪时间步
- ✅ 每步必须训练神经网络
- ✅ Spotter阶段初始化

---

### 策略3：FixedPointSolver - 不动点迭代（新）

```python
# 主循环特征
U = initial_guess()  # 零初值或给定初值
for iter in range(max_iterations):
    # 关键：非线性项 N(U) 固定为上一步的 U
    # 只求解线性系统，不使用 time_step()

    # 核心：固定非线性项，求解线性系统
    coeffs = fitter.fit(U_current=U)  # N(U) 用 U 计算后固定

    # 构造新解
    U_new, U_seg = fitter.construct(data, model, coeffs)

    # 收敛检查
    residual = ||U_new - U|| / (||U|| + eps)
    if residual < tol:
        break  # 收敛终止

    U = U_new  # 简单更新，无时间步概念
```

**特征**：
- ❌ **不使用** `fitter.time_step()`
- ❌ **无时间格式**
- ❌ **无时间推进**（既无物理时间也无伪时间）
- ✅ 非线性项 N(U) 固定为上一步值
- ✅ 每步求解线性系统
- ✅ Picard迭代或拟牛顿迭代
- ✅ 收敛终止

---

## 🔍 三种策略的核心区别总结

| 特性 | TimePDE | GeneralPDE | FixedPoint (新) |
|------|---------|------------|-----------------|
| **核心方法** | `fitter.time_step()` | `fitter.time_step()` | `fitter.fit()` |
| **时间推进** | ✅ 物理时间 T | ✅ 伪时间 τ | ❌ 无 |
| **时间格式** | IMEX-RK(2,2,2) | implicit_euler | ❌ 无 |
| **非线性处理** | 每步训练网络 | 每步训练网络 | 固定为 U_old |
| **终止条件** | T >= T_final | residual < tol | residual < tol |
| **自适应步长** | CFL条件 | 基于残差 | ❌ 无步长概念 |
| **初始化** | 给定 U(t=0) | Spotter训练 | 零值或给定 |
| **Fitter需求** | TimePDEFitter | GeneralPDEFitter | **新的 FixedPointFitter** |

---

## 🏗️ 架构设计方案

### 方案A：平行设计（推荐）

```
NonlinearPDESolver (base)
    ├── TimePDESolver
    ├── GeneralPDESolver
    └── FixedPointSolver (新)
```

**理由**：
- TimePDE 和 GeneralPDE 虽然都用 `time_step()`，但主循环逻辑差异很大
- FixedPoint 完全不同的求解策略
- 三者平行继承，职责清晰

### 方案B：中间层设计（不推荐）

```
NonlinearPDESolver (base)
    ├── TimeSteppingSolver (中间层)
    │   ├── TimePDESolver
    │   └── GeneralPDESolver
    └── FixedPointSolver
```

**问题**：
- TimePDE 和 GeneralPDE 的主循环差异仍然很大（while vs for, T_final vs convergence）
- 中间层能提供的共性很少
- 增加复杂度

**结论**：采用**方案A（平行设计）**

---

## 📦 完整继承层次图

```
src/
├── abstract_class/
│   ├── config/
│   │   ├── base_config.py: BaseConfig
│   │   ├── base_pde_config.py: BasePDEConfig
│   │   │   ├── TimePDEConfig
│   │   │   ├── GeneralPDEConfig
│   │   │   └── FixedPointConfig (新)
│   │   │
│   │   └── base_data.py: BaseDataGenerator
│   │       └── TimePDEDataGenerator
│   │           └── GeneralPDEDataGenerator
│   │
│   ├── base_fitter.py: BaseDeepPolyFitter
│   │   ├── TimePDEFitter (提供 time_step)
│   │   │   └── GeneralPDEFitter (复用 time_step)
│   │   │
│   │   └── FixedPointFitter (新 - 不提供 time_step)
│   │
│   ├── base_net.py: BaseNet
│   │   ├── TimePDENet
│   │   ├── GeneralPDENet
│   │   └── FixedPointNet (新? 或复用现有)
│   │
│   └── config/base_visualize.py: BaseVisualizer
│       ├── TimePDEVisualizer
│       ├── GeneralPDEVisualizer
│       └── FixedPointVisualizer (新)
│
└── problem_solvers/
    └── nonlinear_solver/
        ├── __init__.py
        ├── base_solver.py: NonlinearPDESolver (抽象基类)
        │   ├── TimePDESolver
        │   ├── GeneralPDESolver
        │   └── FixedPointSolver (新)
        │
        └── utils/
            └── solver_utils.py (共享工具函数)
```

---

## 💻 NonlinearPDESolver 基类设计

遵循 `abstract_class` 的三种方法模式：
1. **具体方法** (Concrete Methods): 完整实现的共享功能
2. **抽象方法** (Abstract Methods): 必须由子类实现
3. **钩子方法** (Hook Methods): 可选覆盖的扩展点

### 完整类定义

```python
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
import scipy.io
import os


class NonlinearPDESolver(ABC):
    """非线性PDE求解器抽象基类

    支持三种求解策略：
    1. TimePDESolver - 物理时间演化（固定终止时间）
    2. GeneralPDESolver - 伪时间步稳态（收敛迭代）
    3. FixedPointSolver - 不动点迭代（无时间步）

    设计模式：
    - 具体方法：组件初始化、参考解加载、工具函数
    - 抽象方法：solve(), 工厂方法
    - 钩子方法：求解器特定初始化
    """

    # ==================== 具体方法 (Concrete Methods) ====================

    def __init__(self, config=None, case_dir=None):
        """统一的初始化流程"""
        self._initialize_config(config, case_dir)
        self._initialize_components()
        self._solver_specific_init()  # 钩子方法

    def _initialize_config(self, config, case_dir):
        """配置初始化 - 100%相同的逻辑"""
        if config is not None:
            self.config = config
        elif case_dir is not None:
            self.config = self._create_config(case_dir)  # 抽象方法
        else:
            raise ValueError("Either config object or case_dir must be provided")

        self.case_dir = case_dir if case_dir else getattr(self.config, "case_dir", None)

        # 导出配置文件
        if hasattr(self.config, "export_to_json"):
            self.config.export_to_json("config.json")

    def _initialize_components(self):
        """组件初始化 - 使用工厂方法模式"""
        # Data generator
        self.datagen = self._create_data_generator()  # 抽象方法

        # Generate training and test data
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")

        # Neural network model
        self.model = self._create_model().to(self.config.device)  # 抽象方法

        # Visualizer
        self.visualizer = self._create_visualizer()  # 抽象方法

        # Fitter with optional time scheme
        time_scheme = getattr(self.config, 'time_scheme', self._get_default_time_scheme())
        self.fitter = self._create_fitter(time_scheme)  # 抽象方法

        # Load reference solution
        self.reference_solution = self._load_reference_solution()

        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)

    def _load_reference_solution(self) -> Optional[Dict]:
        """加载参考解 - 通用文件处理 + 维度特定处理"""
        if not hasattr(self.config, 'reference_solution') or not self.config.reference_solution:
            return None

        ref_path = self.config.reference_solution

        # 相对路径转绝对路径
        if not os.path.isabs(ref_path):
            case_dir = getattr(self.config, 'case_dir', os.getcwd())
            ref_path = os.path.join(case_dir, ref_path)

        # .m 文件转 .mat 文件
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

            # 通用：提取 x 坐标
            if 'x' in data:
                x_ref = data['x'].flatten()
            else:
                print("Warning: 'x' not found in reference solution file")
                return None

            # 委托给子类处理维度特定逻辑
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
            # 线性插值
            U_test[:, eq_idx] = griddata(
                x_train, U_train[:, eq_idx], x_test,
                method='linear', fill_value=np.nan
            )

            # NaN 填充
            nan_mask = np.isnan(U_test[:, eq_idx])
            if np.any(nan_mask):
                U_test[nan_mask, eq_idx] = griddata(
                    x_train, U_train[:, eq_idx], x_test[nan_mask],
                    method='nearest'
                )

        print(f"  Interpolated from {x_train.shape[0]} training points to {x_test.shape[0]} test points")
        return U_test

    # ==================== 抽象方法 (Abstract Methods) ====================

    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """主求解方法 - 三种策略完全不同

        TimePDESolver: 固定时间演化循环
        GeneralPDESolver: 伪时间步收敛迭代
        FixedPointSolver: 不动点迭代（无时间步）

        Returns:
            Dict: 包含 solution, coefficients, convergence_info 等
        """
        pass

    @abstractmethod
    def _create_config(self, case_dir: str):
        """创建求解器特定的配置对象

        Returns:
            TimePDEConfig | GeneralPDEConfig | FixedPointConfig
        """
        pass

    @abstractmethod
    def _create_data_generator(self):
        """创建数据生成器

        Returns:
            TimePDEDataGenerator | GeneralPDEDataGenerator | FixedPointDataGenerator
        """
        pass

    @abstractmethod
    def _create_model(self):
        """创建神经网络模型

        Returns:
            TimePDENet | GeneralPDENet | FixedPointNet
        """
        pass

    @abstractmethod
    def _create_fitter(self, time_scheme: Optional[str]):
        """创建Fitter对象

        Args:
            time_scheme: 时间格式名称
                - TimePDE: "imex_rk_222", "imex_1st"
                - GeneralPDE: "implicit_euler"
                - FixedPoint: None (不需要)

        Returns:
            TimePDEFitter | GeneralPDEFitter | FixedPointFitter
        """
        pass

    @abstractmethod
    def _create_visualizer(self):
        """创建可视化器

        Returns:
            TimePDEVisualizer | GeneralPDEVisualizer | FixedPointVisualizer
        """
        pass

    @abstractmethod
    def _get_default_time_scheme(self) -> Optional[str]:
        """获取默认时间格式

        Returns:
            TimePDE: "imex_rk_222"
            GeneralPDE: "implicit_euler"
            FixedPoint: None
        """
        pass

    @abstractmethod
    def _process_reference_solution(self, data: Dict, x_ref: np.ndarray,
                                   mat_file: str) -> Dict:
        """处理参考解数据 - 维度特定逻辑

        Args:
            data: MATLAB数据字典
            x_ref: 空间坐标
            mat_file: 文件路径

        Returns:
            TimePDE: {'x_ref', 't_ref', 'u_ref', 'interpolators'}
            GeneralPDE: {'x_ref', 'y_ref', 'u_ref', 'y_label'}
            FixedPoint: {'x_ref', 'u_ref'}
        """
        pass

    # ==================== 钩子方法 (Hook Methods) ====================

    def _solver_specific_init(self):
        """求解器特定初始化 - 子类可选覆盖

        用途：
        - TimePDE: 初始化 loss_history
        - GeneralPDE: 初始化 Spotter 配置
        - FixedPoint: 初始化迭代参数
        """
        pass

    def _postprocess_results(self, **kwargs):
        """后处理和可视化 - 子类实现

        TimePDE: 时间演化动画、spacetime对比
        GeneralPDE: 收敛历史、残差曲线
        FixedPoint: 迭代收敛图
        """
        pass
```

---

## 🔧 三种Solver的实现框架

### 1. TimePDESolver

```python
# src/problem_solvers/nonlinear_solver/time_pde_solver.py

from .base_solver import NonlinearPDESolver
from src.problem_solvers.time_pde_solver.core.fitter import TimePDEFitter
from src.problem_solvers.time_pde_solver.core.net import TimePDENet
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig
from src.problem_solvers.time_pde_solver.utils.visualize import TimePDEVisualizer
from src.problem_solvers.time_pde_solver.utils.data import TimePDEDataGenerator


class TimePDESolver(NonlinearPDESolver):
    """物理时间演化求解器"""

    # ==================== 实现抽象方法 ====================

    def _create_config(self, case_dir: str):
        return TimePDEConfig(case_dir=case_dir)

    def _create_data_generator(self):
        return TimePDEDataGenerator(self.config)

    def _create_model(self):
        return TimePDENet(self.config)

    def _create_fitter(self, time_scheme: str):
        return TimePDEFitter(
            config=self.config,
            data=self.data_train,
            time_scheme=time_scheme
        )

    def _create_visualizer(self):
        return TimePDEVisualizer(self.config)

    def _get_default_time_scheme(self) -> str:
        return 'imex_rk_222'

    def _process_reference_solution(self, data, x_ref, mat_file):
        """时间演化：提取 t_ref, 创建插值器"""
        # 提取时间坐标
        if 't' in data:
            t_ref = data['t'].flatten()
        else:
            print("Warning: 't' not found in reference solution")
            return None

        # 提取解
        if 'usol' in data:
            u_ref = data['usol']
        elif 'u' in data:
            u_ref = data['u']
        else:
            print("Warning: Solution data not found")
            return None

        # 创建插值器（用于任意时刻的参考解）
        from scipy.interpolate import interp1d
        interpolators = {}
        if u_ref.ndim == 2:  # (n_time, n_space)
            for x_idx in range(len(x_ref)):
                interpolators[x_idx] = interp1d(
                    t_ref, u_ref[:, x_idx],
                    kind='cubic', fill_value='extrapolate'
                )

        return {
            'x_ref': x_ref,
            't_ref': t_ref,
            'u_ref': u_ref,
            'interpolators': interpolators,
            'file_path': mat_file
        }

    # ==================== 覆盖钩子方法 ====================

    def _solver_specific_init(self):
        """时间演化特定初始化"""
        self.loss_history = []
        self.visualizer.reset_animation_data()

    # ==================== 主求解方法 ====================

    def solve(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """物理时间演化主循环"""
        print("\n" + "="*70)
        print("Time-Dependent PDE Solver")
        print("="*70)

        # 初始化
        T = 0.0
        dt = self.config.dt
        U = self.data_train['U0']  # 初始条件
        U_seg = self.fitter.global_to_segments(U)
        coeffs = None

        # 初始化 fitter
        self.fitter.fitter_init(self.model)

        # 时间演化循环
        iter = 0
        while T < self.config.T_final:
            # 神经网络训练（可选）
            if self._should_train_network(iter):
                self._train_neural_network_step(iter, dt, U)

            # 时间步推进
            U, U_seg, coeffs = self.fitter.time_step(
                U, U_seg, dt, coeffs,
                current_time=T, step=iter
            )

            # 更新时间
            T += dt
            iter += 1

            print(f"Step {iter}: T = {T:.6f}, dt = {dt:.6e}")

        # 后处理
        self._postprocess_results(U, U_seg, self.model, coeffs)

        return U, U_seg, self.model, coeffs

    def _should_train_network(self, iter: int) -> bool:
        """判断是否需要训练网络"""
        spotter_skip = getattr(self.config, 'spotter_skip', 1)
        return (iter % spotter_skip == 0)

    def _train_neural_network_step(self, iter: int, dt: float, U_current: np.ndarray):
        """训练神经网络一步"""
        print(f"  Training neural network at iteration {iter}...")

        # 准备GPU数据
        data_GPU = self.model.prepare_gpu_data(self.data_train, U_current)

        # 训练
        self.model.train()
        self.model.train_net(self.data_train, self.model, data_GPU, dt=dt)

        # 重新初始化fitter
        self.model.eval()
        self.fitter.fitter_init(self.model)
```

### 2. GeneralPDESolver

```python
# src/problem_solvers/nonlinear_solver/general_pde_solver.py

from .base_solver import NonlinearPDESolver
from src.problem_solvers.general_pde_solver.core.fitter import GeneralPDEFitter
from src.problem_solvers.general_pde_solver.core.net import GeneralPDENet
from src.problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.utils.visualize import GeneralPDEVisualizer
from src.problem_solvers.general_pde_solver.utils.data import GeneralPDEDataGenerator


class GeneralPDESolver(NonlinearPDESolver):
    """伪时间步稳态求解器"""

    # ==================== 实现抽象方法 ====================

    def _create_config(self, case_dir: str):
        return GeneralPDEConfig(case_dir=case_dir)

    def _create_data_generator(self):
        return GeneralPDEDataGenerator(self.config)

    def _create_model(self):
        return GeneralPDENet(self.config)

    def _create_fitter(self, time_scheme: str):
        return GeneralPDEFitter(
            config=self.config,
            data=self.data_train,
            time_scheme=time_scheme,
            visualizer=self.visualizer
        )

    def _create_visualizer(self):
        return GeneralPDEVisualizer(self.config)

    def _get_default_time_scheme(self) -> str:
        return 'implicit_euler'

    def _process_reference_solution(self, data, x_ref, mat_file):
        """稳态：提取 y_ref (或 t 作为第二空间维度)"""
        # 检查是否有第二维坐标
        y_ref = None
        y_label = 'y'

        if 'y' in data:
            y_ref = data['y'].flatten()
        elif 't' in data:
            y_ref = data['t'].flatten()
            y_label = 't'  # 时间作为空间维度

        # 提取解
        if 'usol' in data:
            u_ref = data['usol']
        elif 'u' in data:
            u_ref = data['u']
        else:
            print("Warning: Solution data not found")
            return None

        # 展平2D解
        if u_ref.ndim == 2 and y_ref is not None:
            u_ref = u_ref.flatten()

        result = {
            'x_ref': x_ref,
            'u_ref': u_ref,
            'file_path': mat_file
        }

        if y_ref is not None:
            result['y_ref'] = y_ref
            result['y_label'] = y_label

        return result

    # ==================== 主求解方法 ====================

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict:
        """伪时间步迭代主循环"""
        print("\n" + "="*70)
        print("General PDE Solver - Pseudo-Time Stepping")
        print("="*70)

        # Step 1: 初始化
        init_method = getattr(self.config, 'initial_condition', {}).get('method', 'spotter')

        if init_method == 'spotter':
            U_init = self._spotter_stage()
        elif init_method == 'given':
            U_init = self._get_initial_from_config()
        elif init_method == 'zero':
            n_points = self.data_train['x'].shape[0]
            U_init = np.zeros((n_points, self.config.n_eqs))
        else:
            raise ValueError(f"Unknown initial_condition method: {init_method}")

        # Step 2: Fitter 初始化
        self.fitter.fitter_init(self.model)

        # Step 3: 伪时间步迭代
        tau = 0.0
        dtau = getattr(self.config, 'dt_init', 1.0)
        U = U_init
        U_seg = self.fitter.global_to_segments(U)
        coeffs = None

        max_iter = getattr(self.config, 'max_iterations', 1000)
        conv_tol = getattr(self.config, 'convergence_tol', 1e-6)

        for iter in range(max_iter):
            # 训练神经网络
            self._train_neural_network_step(iter, dtau, U)

            # 伪时间步推进
            U_prev = U.copy()
            U, U_seg, coeffs = self.fitter.time_step(
                U, U_seg, dtau, coeffs,
                current_time=tau, step=iter
            )

            # 收敛检查
            residual = np.linalg.norm(U - U_prev) / (np.linalg.norm(U_prev) + 1e-10)
            print(f"Iteration {iter}: tau = {tau:.6f}, dtau = {dtau:.6e}, residual = {residual:.6e}")

            if residual < conv_tol:
                print(f"✓ Converged at iteration {iter}")
                converged = True
                break

            # 自适应伪时间步
            dtau = self._adapt_pseudo_timestep(dtau, residual)
            tau += dtau
        else:
            print(f"✗ Max iterations reached without convergence")
            converged = False

        # 后处理
        conv_info = {
            "converged": converged,
            "iterations": iter,
            "final_residual": residual
        }

        self._postprocess(U, coeffs, conv_info)

        return {
            "solution": U,
            "coefficients": coeffs,
            "convergence_info": conv_info
        }

    def _spotter_stage(self):
        """Spotter网络初始化"""
        print("  Using Spotter network for initialization...")

        spotter_model = SpotterNet(self.config).to(self.config.device)
        spotter_config = getattr(self.config, 'initial_condition', {}).get('spotter_config', {})

        spotter_model.train_spotter(
            data=self.data_train,
            n_epochs=spotter_config.get('epochs', 1000),
            lr_init=spotter_config.get('learning_rate', 1e-2),
            n_pde_points=spotter_config.get('n_pde_points', 10000)
        )

        # 转移权重
        self.model.load_state_dict(spotter_model.state_dict())

        # 提取初值
        with torch.no_grad():
            x_tensor = torch.tensor(self.data_train['x'], dtype=torch.float64,
                                   device=self.config.device)
            _, U_init = self.model(x_tensor)
            U_init = U_init.cpu().numpy()

        return U_init

    def _adapt_pseudo_timestep(self, dtau, residual):
        """基于残差的自适应伪时间步"""
        dt_max = getattr(self.config, 'dt_max', 10.0)
        dt_min = getattr(self.config, 'dt_min', 0.01)

        if residual > 0.1:
            dtau_new = dt_max
        elif residual > 0.01:
            dtau_new = dt_max * 0.5
        elif residual > 1e-4:
            dtau_new = dt_max * 0.2
        else:
            dtau_new = dt_min

        return dtau_new
```

### 3. FixedPointSolver（新）

```python
# src/problem_solvers/nonlinear_solver/fixed_point_solver.py

from .base_solver import NonlinearPDESolver
from src.problem_solvers.fixed_point_solver.core.fitter import FixedPointFitter
from src.problem_solvers.fixed_point_solver.core.net import FixedPointNet
from src.problem_solvers.fixed_point_solver.utils.config import FixedPointConfig
from src.problem_solvers.fixed_point_solver.utils.visualize import FixedPointVisualizer
from src.problem_solvers.fixed_point_solver.utils.data import FixedPointDataGenerator


class FixedPointSolver(NonlinearPDESolver):
    """不动点迭代求解器 - 非线性项固定为上一步值"""

    # ==================== 实现抽象方法 ====================

    def _create_config(self, case_dir: str):
        return FixedPointConfig(case_dir=case_dir)

    def _create_data_generator(self):
        return FixedPointDataGenerator(self.config)

    def _create_model(self):
        return FixedPointNet(self.config)

    def _create_fitter(self, time_scheme: Optional[str]):
        """FixedPoint不需要time_scheme"""
        return FixedPointFitter(
            config=self.config,
            data=self.data_train
        )

    def _create_visualizer(self):
        return FixedPointVisualizer(self.config)

    def _get_default_time_scheme(self) -> Optional[str]:
        return None  # FixedPoint不使用时间格式

    def _process_reference_solution(self, data, x_ref, mat_file):
        """稳态问题：只提取 x_ref 和 u_ref"""
        if 'usol' in data:
            u_ref = data['usol'].flatten()
        elif 'u' in data:
            u_ref = data['u'].flatten()
        else:
            print("Warning: Solution data not found")
            return None

        return {
            'x_ref': x_ref,
            'u_ref': u_ref,
            'file_path': mat_file
        }

    # ==================== 主求解方法 ====================

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict:
        """不动点迭代主循环"""
        print("\n" + "="*70)
        print("Fixed-Point Iteration Solver")
        print("="*70)

        # 初始化
        if U_init is None:
            # 零初值
            n_points = self.data_train['x'].shape[0]
            U = np.zeros((n_points, self.config.n_eqs))
        else:
            U = U_init

        # Fitter 初始化
        self.fitter.fitter_init(self.model)

        # 不动点迭代
        max_iter = getattr(self.config, 'max_iterations', 100)
        conv_tol = getattr(self.config, 'convergence_tol', 1e-6)

        for iter in range(max_iter):
            # 关键：非线性项 N(U) 固定为当前的 U
            # fitter.fit() 内部会用 U 计算 N(U) 并固定它
            coeffs = self.fitter.fit(U_current=U)

            # 构造新解
            U_new, U_seg = self.fitter.construct(
                self.data_train, self.model, coeffs
            )

            # 收敛检查
            residual = np.linalg.norm(U_new - U) / (np.linalg.norm(U) + 1e-10)
            print(f"Iteration {iter}: residual = {residual:.6e}")

            if residual < conv_tol:
                print(f"✓ Converged at iteration {iter}")
                converged = True
                U = U_new
                break

            # 更新解
            U = U_new
        else:
            print(f"✗ Max iterations reached without convergence")
            converged = False

        # 后处理
        conv_info = {
            "converged": converged,
            "iterations": iter,
            "final_residual": residual
        }

        self._postprocess(U, coeffs, conv_info)

        return {
            "solution": U,
            "coefficients": coeffs,
            "convergence_info": conv_info
        }
```

---

## 🔩 Fitter 层支持：FixedPointFitter

### FixedPointFitter 设计

```python
# src/problem_solvers/fixed_point_solver/core/fitter.py

from src.abstract_class.base_fitter import BaseDeepPolyFitter
import numpy as np
from typing import Dict


class FixedPointFitter(BaseDeepPolyFitter):
    """不动点迭代的Fitter - 非线性项固定为给定值

    与 TimePDEFitter 的区别：
    - 不提供 time_step() 方法
    - 提供 fit(U_current) 方法，固定 N(U) = N(U_current)
    - 只求解线性系统
    """

    def fit(self, U_current: np.ndarray, **kwargs) -> np.ndarray:
        """固定非线性项，求解线性系统

        求解：L1*U + L2*U = F - N(U_current)

        其中 N(U_current) 在构建 Jacobian 时固定为给定的 U_current 的值

        Args:
            U_current: 当前解，用于计算非线性项 N(U_current)

        Returns:
            coeffs: 解的系数 (ns, n_eqs, dgN)
        """
        # 将全局 U 转换为段级
        U_segments = self.global_to_segments(U_current)

        # 构建 Jacobian，N(U_current) 作为右端项
        L, r = self._build_jacobian(U_current=U_segments, **kwargs)

        # 求解线性系统
        coeffs_flat = self.solver.solve(L, r, method="auto")[0]

        # 重塑为段级系数
        coeffs = coeffs_flat.reshape((self.ns, self.config.n_eqs, self.dgN))

        return coeffs

    def _build_segment_jacobian(self, segment_idx: int, U_current=None, **kwargs):
        """构建单个段的Jacobian

        关键：非线性项 N(U) 固定为 U_current[segment_idx]

        线性系统：
            (L1 + L2) * coeffs = F - N(U_current)
        """
        # 获取预编译的特征和算子
        features = self._features[segment_idx]
        linear_ops = self._linear_operators[segment_idx]
        nonlinear_funcs = self._nonlinear_functions[segment_idx]

        n_points = len(self.data["x_segments_norm"][segment_idx])
        ne = self.config.n_eqs
        dgN = self.dgN

        # 线性算子矩阵
        L1_mat = linear_ops["L1"]  # (n_points*ne, dgN*ne)
        L2_mat = linear_ops["L2"]  # (n_points*ne, dgN*ne)

        # Jacobian = L1 + L2
        J = L1_mat + L2_mat

        # 右端项：F - N(U_current)
        F_vec = nonlinear_funcs["F"](coeffs=None)  # 源项（不依赖系数）

        # 关键：固定非线性项为 U_current
        if U_current is not None and segment_idx < len(U_current):
            U_seg_current = U_current[segment_idx]  # (n_points, ne)

            # 从 U 反推系数（近似）
            # 或者直接用 U 计算 N(U)
            # 这里简化处理：直接用 U 作为非线性项的值
            N_vec = U_seg_current.flatten()  # (n_points*ne,)
        else:
            N_vec = np.zeros(n_points * ne)

        # 右端项 r = F - N(U_current)
        r = F_vec - N_vec

        return J, r
```

### FixedPointFitter 继承层次

```
BaseDeepPolyFitter
    ├── TimePDEFitter (提供 time_step)
    │   └── GeneralPDEFitter
    │
    └── FixedPointFitter (新 - 提供 fit, 不提供 time_step)
```

**设计理由**：
- FixedPointFitter 不需要时间步推进，因此不继承 TimePDEFitter
- 直接继承 BaseDeepPolyFitter，复用 92% 的核心功能
- 实现自己的 `fit()` 方法，固定非线性项

---

## 📊 对比：abstract_class 的设计模式

| 设计元素 | abstract_class | nonlinear_solver (本设计) |
|----------|----------------|---------------------------|
| **基类** | `BaseConfig`, `BaseDeepPolyFitter`, `BaseNet` | `NonlinearPDESolver` |
| **具体方法** | `_precompile_all_operators()`, `_add_constraints()` | `_initialize_config()`, `_load_reference_solution()` |
| **抽象方法** | `_build_segment_jacobian()` | `solve()`, `_create_config()`, `_create_fitter()` |
| **钩子方法** | `_solver_specific_init()` (BasePDEConfig) | `_solver_specific_init()`, `_postprocess_results()` |
| **工厂方法** | `create_time_scheme()`, `create_operator_factory()` | `_create_config()`, `_create_fitter()`, ... |
| **继承深度** | 2-3层 | 2层（NonlinearPDESolver → 具体Solver） |
| **代码复用** | 92% | 目标：60-70% |

**设计一致性**：
✅ 遵循 abstract_class 的三种方法模式
✅ 使用工厂方法模式创建组件
✅ 钩子方法支持可选扩展
✅ 抽象方法强制子类实现核心逻辑

---

## 🗂️ 目录结构

```
src/problem_solvers/
├── nonlinear_solver/
│   ├── __init__.py
│   ├── base_solver.py              # NonlinearPDESolver 基类
│   └── utils/
│       └── solver_utils.py         # 共享工具函数
│
├── time_pde_solver/
│   ├── solver.py → 继承 NonlinearPDESolver
│   ├── core/
│   │   ├── fitter.py               # TimePDEFitter
│   │   └── net.py                  # TimePDENet
│   └── utils/
│       ├── config.py               # TimePDEConfig
│       ├── data.py                 # TimePDEDataGenerator
│       └── visualize.py            # TimePDEVisualizer
│
├── general_pde_solver/
│   ├── solver.py → 继承 NonlinearPDESolver
│   ├── core/
│   │   ├── fitter.py               # GeneralPDEFitter ← TimePDEFitter
│   │   ├── net.py                  # GeneralPDENet
│   │   └── spotter_init.py         # SpotterNet
│   └── utils/
│       ├── config.py               # GeneralPDEConfig
│       ├── data.py                 # GeneralPDEDataGenerator
│       └── visualize.py            # GeneralPDEVisualizer
│
└── fixed_point_solver/              # 新的求解器
    ├── solver.py → 继承 NonlinearPDESolver
    ├── core/
    │   ├── fitter.py               # FixedPointFitter ← BaseDeepPolyFitter
    │   └── net.py                  # FixedPointNet (复用或新建)
    └── utils/
        ├── config.py               # FixedPointConfig
        ├── data.py                 # FixedPointDataGenerator (可能复用)
        └── visualize.py            # FixedPointVisualizer
```

---

## 🚀 实施路线图

### 阶段1：创建基础架构（2-3小时）

1. **创建目录结构**：
   ```bash
   mkdir -p src/problem_solvers/nonlinear_solver/utils
   mkdir -p src/problem_solvers/fixed_point_solver/core
   mkdir -p src/problem_solvers/fixed_point_solver/utils
   ```

2. **实现 NonlinearPDESolver 基类**：
   - `src/problem_solvers/nonlinear_solver/base_solver.py`
   - 包含所有具体方法、抽象方法、钩子方法

3. **创建 FixedPointFitter**：
   - `src/problem_solvers/fixed_point_solver/core/fitter.py`
   - 继承 `BaseDeepPolyFitter`
   - 实现 `fit(U_current)` 方法

### 阶段2：迁移 TimePDESolver（1-2小时）

1. **修改 TimePDESolver**：
   - 继承 `NonlinearPDESolver`
   - 删除重复方法（`_initialize_config`, `_load_reference_solution`, 等）
   - 实现抽象方法（工厂方法）
   - 保留 `solve()` 主循环

2. **测试**：
   - 运行现有 time_pde 测试用例
   - 确保结果一致

### 阶段3：迁移 GeneralPDESolver（1-2小时）

1. **修改 GeneralPDESolver**：
   - 继承 `NonlinearPDESolver`
   - 删除重复方法
   - 实现抽象方法
   - 保留 `solve()` 和 `_spotter_stage()`

2. **测试**：
   - 运行现有 general_pde 测试用例
   - 确保收敛性和精度

### 阶段4：实现 FixedPointSolver（3-4小时）

1. **实现 FixedPointSolver**：
   - 继承 `NonlinearPDESolver`
   - 实现 `solve()` 主循环
   - 实现工厂方法

2. **实现配套组件**：
   - FixedPointConfig
   - FixedPointVisualizer
   - FixedPointDataGenerator（可能复用 TimePDEDataGenerator）

3. **测试**：
   - 创建简单测试用例（如非线性Poisson方程）
   - 验证收敛性

### 阶段5：文档和优化（1小时）

1. **更新文档**：
   - 更新 README
   - 添加使用示例

2. **代码优化**：
   - 添加类型注解
   - 改进注释

---

## ✅ 总结

### 关键设计决策

1. **平行继承架构**：三种 Solver 平行继承 `NonlinearPDESolver`
   - 清晰的职责划分
   - 避免不必要的继承复杂度

2. **Fitter 层分离**：
   - `TimePDEFitter` 和 `GeneralPDEFitter` 保持现有继承关系（复用 time_step）
   - `FixedPointFitter` 独立继承 `BaseDeepPolyFitter`（不需要 time_step）

3. **遵循 abstract_class 设计模式**：
   - 具体方法：共享功能（初始化、参考解加载、工具函数）
   - 抽象方法：核心差异（solve, 工厂方法）
   - 钩子方法：可选扩展（特定初始化、后处理）

4. **代码复用**：
   - Solver 层：60-70% 代码复用
   - Fitter 层：92% 代码复用（已有）
   - 总体维护成本大幅降低

### 三种策略的本质区别

| 策略 | 核心特征 | 适用场景 |
|------|---------|---------|
| **TimePDE** | 物理时间演化，固定终止时间 | 时间演化PDE（热方程、波动方程） |
| **GeneralPDE** | 伪时间步，收敛至稳态 | 稳态PDE，时间作为空间维度 |
| **FixedPoint** | 不动点迭代，无时间步 | 稳态非线性PDE，Picard迭代 |

### 实施优先级

1. ✅ **高优先级**：NonlinearPDESolver 基类 + TimePDESolver 迁移
2. ✅ **中优先级**：GeneralPDESolver 迁移
3. 🔄 **低优先级**：FixedPointSolver 实现（新功能）

---

## 🎯 下一步行动

请确认：
1. 架构设计是否符合预期？
2. 三种策略的划分是否合理？
3. FixedPointFitter 的设计是否满足需求？
4. 是否开始实施阶段1（创建基础架构）？
