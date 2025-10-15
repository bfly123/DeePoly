# NonlinearPDESolver 统一基类设计

**日期**: 2025-10-10
**目标**: 设计 NonlinearPDESolver 基类，充分提取三种 Solver 的共性，同时保持它们的独立性

---

## 🔍 三种 Solver 的深度共性分析

### 数学本质联系

**关键洞察**：伪时间步和不动点迭代的本质联系

```
伪时间步（隐式Euler）：
    (U^{n+1} - U^n)/Δτ + L(U^{n+1}) + N(U^{n+1}) = F

    ↓ 移项

    [(1/Δτ)·I + L]·U^{n+1} + N(U^{n+1}) = F + (1/Δτ)·I·U^n
                ↑ 阻尼项

不动点迭代（Δτ → ∞）：
    L(U^{n+1}) + N(U^n) = F
    ↑ 无阻尼项，非线性项固定
```

**结论**：
- GeneralPDE（伪时间步）= 不动点迭代 + 阻尼正则化 + 神经网络训练
- FixedPoint = Δτ → ∞ 的极限情况

---

## 📊 三种 Solver 的共性矩阵

| 特性 | TimePDE | GeneralPDE | FixedPoint | 共性程度 |
|------|---------|------------|------------|----------|
| **初始化流程** | ✅ | ✅ | ✅ | **100%** |
| **组件创建** | ✅ | ✅ | ✅ | **100%** (工厂模式) |
| **参考解加载** | ✅ | ✅ | ✅ | **90%** (文件处理相同) |
| **迭代循环** | while T<T_final | for iter (收敛) | for iter (收敛) | **66%** |
| **收敛检查** | ❌ | ✅ residual<tol | ✅ residual<tol | **66%** |
| **残差计算** | ❌ | ✅ \|\|U^{n+1}-U^n\|\| | ✅ \|\|U^{n+1}-U^n\|\| | **66%** |
| **神经网络训练** | ✅ 可选 | ✅ 每步 | ⚠️ 可选/不需要 | **100%** (框架相同) |
| **Fitter调用** | time_step | time_step | fit/time_step | **66%** |
| **后处理** | ✅ | ✅ | ✅ | **80%** |
| **插值到测试网格** | ✅ | ✅ | ✅ | **100%** |
| **GPU/CPU转换** | ✅ | ✅ | ✅ | **100%** |

### 共性识别

**1. 100% 共性（具体方法）**：
- 初始化流程（_initialize_config, _initialize_components）
- 组件工厂方法（_create_config, _create_model, etc.）
- 参考解加载框架（_load_reference_solution）
- GPU/CPU数据转换（_to_numpy_cpu）
- 插值到测试网格（_interpolate_to_test_grid）

**2. 66% 共性（模板方法 + 钩子）**：
- 收敛迭代框架（GeneralPDE 和 FixedPoint 共享）
- 收敛检查逻辑（_check_convergence, _compute_residual）
- 神经网络训练框架（三者可选使用）

**3. 特定差异（抽象方法）**：
- solve() 主循环（三者完全不同）
- _process_reference_solution()（维度特定）

---

## 💻 NonlinearPDESolver 基类设计（增强版）

### 设计原则

1. **最大化代码复用**：提取所有共性到基类
2. **保持灵活性**：子类可以完全覆盖任何方法
3. **提供辅助工具**：收敛检查、残差计算等工具方法
4. **模板方法模式**：提供可选的迭代框架

### 完整类定义

```python
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, Callable
import numpy as np
import torch
import scipy.io
import os


class NonlinearPDESolver(ABC):
    """非线性PDE求解器统一基类

    支持三种求解策略：
    1. TimePDESolver - 物理时间演化（固定终止时间）
    2. GeneralPDESolver - 伪时间步稳态（收敛迭代 + 阻尼）
    3. FixedPointSolver - 不动点迭代（收敛迭代，无阻尼）

    设计模式：
    - 具体方法：100%共性的功能
    - 模板方法：66%共性的框架（可选使用）
    - 抽象方法：特定差异的核心逻辑
    - 钩子方法：可选扩展点
    """

    # ==================== 具体方法 (Concrete Methods) - 100%共性 ====================

    def __init__(self, config=None, case_dir=None):
        """统一的初始化流程"""
        self._initialize_config(config, case_dir)
        self._initialize_components()
        self._solver_specific_init()  # 钩子方法

    def _initialize_config(self, config, case_dir):
        """配置初始化 - 100%相同"""
        if config is not None:
            self.config = config
        elif case_dir is not None:
            self.config = self._create_config(case_dir)  # 抽象方法
        else:
            raise ValueError("Either config object or case_dir must be provided")

        self.case_dir = case_dir if case_dir else getattr(self.config, "case_dir", None)

        if hasattr(self.config, "export_to_json"):
            self.config.export_to_json("config.json")

    def _initialize_components(self):
        """组件初始化 - 工厂方法模式"""
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
        """加载参考解 - 通用文件处理"""
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
        """GPU数据转CPU numpy数组 - 100%共性"""
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
        """从训练点插值到测试网格 - 100%共性"""
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

    # ==================== 模板方法 (Template Methods) - 66%共性框架 ====================

    def _convergence_iteration_loop(
        self,
        U_init: np.ndarray,
        max_iterations: int,
        conv_tol: float,
        iteration_step: Callable,
        compute_residual: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """收敛迭代框架 - GeneralPDE 和 FixedPoint 共享

        提供标准的收敛迭代循环框架，子类提供具体的迭代步骤。

        Args:
            U_init: 初始解
            max_iterations: 最大迭代次数
            conv_tol: 收敛容差
            iteration_step: 迭代步骤函数 (iter, U, **kwargs) -> U_new
            compute_residual: 残差计算函数，默认使用 _compute_residual
            **kwargs: 传递给 iteration_step 的额外参数

        Returns:
            (U_final, convergence_info)
        """
        U = U_init.copy()

        if compute_residual is None:
            compute_residual = self._compute_residual

        for iter in range(max_iterations):
            # 执行一步迭代（由子类提供）
            U_new = iteration_step(iter, U, **kwargs)

            # 计算残差
            residual = compute_residual(U_new, U)

            # 打印信息
            print(f"Iteration {iter}: residual = {residual:.6e}")

            # 收敛检查
            if self._check_convergence(residual, conv_tol):
                print(f"✓ Converged at iteration {iter}")
                return U_new, {
                    "converged": True,
                    "iterations": iter,
                    "final_residual": residual
                }

            # 更新解
            U = U_new

        # 未收敛
        print(f"✗ Max iterations {max_iterations} reached without convergence")
        return U, {
            "converged": False,
            "iterations": max_iterations,
            "final_residual": residual
        }

    def _compute_residual(self, U_new: np.ndarray, U_old: np.ndarray) -> float:
        """计算相对残差 - 标准实现

        residual = ||U_new - U_old|| / (||U_old|| + eps)

        子类可以覆盖此方法使用不同的残差定义。
        """
        return np.linalg.norm(U_new - U_old) / (np.linalg.norm(U_old) + 1e-10)

    def _check_convergence(self, residual: float, tolerance: float) -> bool:
        """检查是否收敛 - 标准实现"""
        return residual < tolerance

    def _train_neural_network_step(
        self,
        iter: int,
        dt: float,
        U_current: np.ndarray,
        force: bool = False
    ):
        """神经网络训练步骤 - 通用框架

        Args:
            iter: 当前迭代次数
            dt: 时间步长（物理时间或伪时间）
            U_current: 当前解
            force: 强制训练（忽略 should_train 检查）
        """
        # 检查是否需要训练
        if not force and not self._should_train_network(iter):
            print(f"  Skipping neural network training at iteration {iter}")
            # 仍然需要重新初始化 fitter（可能模型参数变了）
            self.fitter.fitter_init(self.model)
            return

        print(f"  Training neural network at iteration {iter}...")

        # 准备GPU数据
        data_GPU = self.model.prepare_gpu_data(self.data_train, U_current)

        # 训练
        self.model.train()
        self.model.train_net(self.data_train, self.model, data_GPU, dt=dt)

        # 重新初始化fitter
        self.model.eval()
        final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
        print(f"  Neural network training completed, loss: {final_loss:.8e}")

        self.fitter.fitter_init(self.model)

    def _should_train_network(self, iter: int) -> bool:
        """判断是否需要训练网络 - 默认实现

        子类可以覆盖以实现不同的训练策略：
        - TimePDE: 每 spotter_skip 步训练一次
        - GeneralPDE: 每步都训练
        - FixedPoint: 可能永不训练
        """
        spotter_skip = getattr(self.config, 'spotter_skip', 1)
        return (iter % spotter_skip == 0)

    # ==================== 抽象方法 (Abstract Methods) ====================

    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """主求解方法 - 三种策略完全不同

        TimePDESolver:
            - 固定时间演化循环 while T < T_final
            - 返回 (U, U_seg, model, coeffs)

        GeneralPDESolver:
            - 伪时间步收敛迭代 for iter in range(max_iter)
            - 每步训练网络，time_step(dt有限)
            - 返回 {"solution", "coefficients", "convergence_info"}

        FixedPointSolver:
            - 不动点迭代 for iter in range(max_iter)
            - 不训练网络（或可选），fit() 或 time_step(dt→∞)
            - 返回 {"solution", "coefficients", "convergence_info"}
        """
        pass

    @abstractmethod
    def _create_config(self, case_dir: str):
        """创建求解器特定的配置对象"""
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
        """创建Fitter对象

        Args:
            time_scheme:
                - TimePDE: "imex_rk_222", "imex_1st"
                - GeneralPDE: "implicit_euler"
                - FixedPoint: None 或 "implicit_euler"（如果用time_step）
        """
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
    def _process_reference_solution(
        self,
        data: Dict,
        x_ref: np.ndarray,
        mat_file: str
    ) -> Dict:
        """处理参考解数据 - 维度特定逻辑"""
        pass

    # ==================== 钩子方法 (Hook Methods) ====================

    def _solver_specific_init(self):
        """求解器特定初始化 - 子类可选覆盖"""
        pass

    def _postprocess_results(self, **kwargs):
        """后处理和可视化 - 子类实现"""
        pass

    def _get_initial_guess(self) -> np.ndarray:
        """获取初始猜测值 - 默认零初值

        子类可以覆盖以提供更好的初值：
        - TimePDE: 从 data_train['U0'] 获取
        - GeneralPDE: Spotter 初始化
        - FixedPoint: 零初值或给定初值
        """
        n_points = self.data_train['x'].shape[0]
        n_eqs = self.config.n_eqs
        return np.zeros((n_points, n_eqs))
```

---

## 🔧 三种 Solver 的实现（利用基类框架）

### 1. TimePDESolver - 保持原有逻辑

```python
class TimePDESolver(NonlinearPDESolver):
    """物理时间演化求解器"""

    def solve(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """物理时间演化主循环 - 不使用 _convergence_iteration_loop"""
        print("\n" + "="*70)
        print("Time-Dependent PDE Solver")
        print("="*70)

        # 初始化
        T = 0.0
        dt = self.config.dt
        U = self.data_train['U0']
        U_seg = self.fitter.global_to_segments(U)
        coeffs = None

        self.fitter.fitter_init(self.model)

        # 时间演化循环（不同于收敛迭代）
        iter = 0
        while T < self.config.T_final:
            # 可选训练（利用基类方法）
            self._train_neural_network_step(iter, dt, U)

            # 时间步推进
            U, U_seg, coeffs = self.fitter.time_step(
                U, U_seg, dt, coeffs,
                current_time=T, step=iter
            )

            T += dt
            iter += 1
            print(f"Step {iter}: T = {T:.6f}, dt = {dt:.6e}")

        self._postprocess_results(U, U_seg, self.model, coeffs)
        return U, U_seg, self.model, coeffs

    def _solver_specific_init(self):
        """时间演化特定初始化"""
        self.loss_history = []
        self.visualizer.reset_animation_data()

    # 工厂方法实现...
```

### 2. GeneralPDESolver - 使用基类框架

```python
class GeneralPDESolver(NonlinearPDESolver):
    """伪时间步稳态求解器"""

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict:
        """伪时间步迭代主循环 - 可以选择使用基类框架"""
        print("\n" + "="*70)
        print("General PDE Solver - Pseudo-Time Stepping")
        print("="*70)

        # Step 1: 初始化
        if U_init is None:
            U_init = self._get_initial_guess()

        self.fitter.fitter_init(self.model)

        # Step 2: 迭代参数
        max_iter = getattr(self.config, 'max_iterations', 1000)
        conv_tol = getattr(self.config, 'convergence_tol', 1e-6)
        dtau = getattr(self.config, 'dt_init', 1.0)

        # 方案A：自己实现循环（当前方式）
        U = U_init
        U_seg = self.fitter.global_to_segments(U)
        coeffs = None
        tau = 0.0

        for iter in range(max_iter):
            # 训练神经网络（利用基类方法）
            self._train_neural_network_step(iter, dtau, U, force=True)

            # 伪时间步推进
            U_prev = U.copy()
            U, U_seg, coeffs = self.fitter.time_step(
                U, U_seg, dtau, coeffs,
                current_time=tau, step=iter
            )

            # 收敛检查（利用基类方法）
            residual = self._compute_residual(U, U_prev)
            print(f"Iteration {iter}: tau = {tau:.6f}, dtau = {dtau:.6e}, residual = {residual:.6e}")

            if self._check_convergence(residual, conv_tol):
                print(f"✓ Converged at iteration {iter}")
                converged = True
                break

            # 自适应伪时间步
            dtau = self._adapt_pseudo_timestep(dtau, residual)
            tau += dtau
        else:
            converged = False

        # 方案B：使用基类框架（简化版 - 可选）
        # def iteration_step(iter, U, dtau_current):
        #     self._train_neural_network_step(iter, dtau_current, U, force=True)
        #     U_prev = U.copy()
        #     U_new, _, coeffs = self.fitter.time_step(U, ..., dtau_current, ...)
        #     return U_new
        #
        # U, conv_info = self._convergence_iteration_loop(
        #     U_init, max_iter, conv_tol, iteration_step, dtau=dtau
        # )

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

    def _get_initial_guess(self) -> np.ndarray:
        """覆盖基类方法 - Spotter 初始化"""
        init_method = getattr(self.config, 'initial_condition', {}).get('method', 'spotter')

        if init_method == 'spotter':
            return self._spotter_stage()
        elif init_method == 'given':
            return self._get_initial_from_config()
        elif init_method == 'zero':
            return super()._get_initial_guess()
        else:
            raise ValueError(f"Unknown initial_condition method: {init_method}")

    def _should_train_network(self, iter: int) -> bool:
        """覆盖基类方法 - GeneralPDE 每步都训练"""
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

    # _spotter_stage(), 工厂方法等...
```

### 3. FixedPointSolver - 充分利用基类框架

```python
class FixedPointSolver(NonlinearPDESolver):
    """不动点迭代求解器 - 充分利用基类框架"""

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict:
        """不动点迭代主循环 - 使用基类框架"""
        print("\n" + "="*70)
        print("Fixed-Point Iteration Solver")
        print("="*70)

        # 初始化
        if U_init is None:
            U_init = self._get_initial_guess()

        self.fitter.fitter_init(self.model)

        # 迭代参数
        max_iter = getattr(self.config, 'max_iterations', 100)
        conv_tol = getattr(self.config, 'convergence_tol', 1e-6)

        # 定义迭代步骤
        def iteration_step(iter: int, U: np.ndarray) -> np.ndarray:
            """单步不动点迭代

            关键：N(U) 固定为当前的 U
            """
            # 固定非线性项，求解线性系统
            coeffs = self.fitter.fit(U_current=U)

            # 构造新解
            U_new, _ = self.fitter.construct(self.data_train, self.model, coeffs)

            return U_new

        # 使用基类提供的收敛迭代框架
        U_final, conv_info = self._convergence_iteration_loop(
            U_init=U_init,
            max_iterations=max_iter,
            conv_tol=conv_tol,
            iteration_step=iteration_step
        )

        self._postprocess(U_final, None, conv_info)

        return {
            "solution": U_final,
            "coefficients": None,  # 可以在 iteration_step 中保存
            "convergence_info": conv_info
        }

    def _should_train_network(self, iter: int) -> bool:
        """覆盖基类方法 - FixedPoint 通常不训练网络"""
        # 可选：根据配置决定是否训练
        train_mode = getattr(self.config, 'fixed_point_train_mode', 'never')

        if train_mode == 'never':
            return False
        elif train_mode == 'first_only':
            return (iter == 0)
        elif train_mode == 'periodic':
            period = getattr(self.config, 'train_period', 10)
            return (iter % period == 0)
        else:
            return False

    # 工厂方法实现...
```

---

## 📊 基类设计总结

### 提供的共性功能

| 功能类别 | 具体方法 | 使用者 |
|---------|---------|--------|
| **初始化** | `_initialize_config`, `_initialize_components` | 三者都用 |
| **工具函数** | `_to_numpy_cpu`, `_interpolate_to_test_grid` | 三者都用 |
| **收敛迭代框架** | `_convergence_iteration_loop` | GeneralPDE, FixedPoint |
| **残差计算** | `_compute_residual`, `_check_convergence` | GeneralPDE, FixedPoint |
| **神经网络训练** | `_train_neural_network_step`, `_should_train_network` | 三者可选用 |
| **参考解加载** | `_load_reference_solution` | 三者都用 |
| **初值获取** | `_get_initial_guess` (钩子) | 三者可覆盖 |

### 代码复用统计

```
具体方法（100%共性）：      ~200行
模板方法（66%共性框架）：    ~100行
抽象方法（接口定义）：       ~50行
钩子方法（可选扩展）：       ~30行
-------------------------------------------
NonlinearPDESolver 基类总计：~380行

各 Solver 需要实现：
- TimePDESolver:     ~250行（不使用迭代框架）
- GeneralPDESolver:  ~200行（可选使用迭代框架）
- FixedPointSolver:  ~150行（充分利用迭代框架）
```

### 设计优势

1. **最大化代码复用**：
   - 100% 共性直接继承使用
   - 66% 共性提供可选框架
   - 总代码复用：~60-70%

2. **保持独立性**：
   - 三个 Solver 仍然是独立的类
   - 各自实现 `solve()` 方法
   - 可以完全覆盖任何方法

3. **灵活性**：
   - 可以选择使用基类框架（如 FixedPoint）
   - 也可以完全自己实现（如 TimePDE）
   - 或部分使用（如 GeneralPDE）

4. **数学直观性**：
   - `_convergence_iteration_loop` 体现了 GeneralPDE 和 FixedPoint 的共性
   - `_train_neural_network_step` 统一了神经网络训练接口
   - 同时保持了它们的独立身份

---

## 🎯 实施建议

### 阶段1：创建增强的 NonlinearPDESolver 基类

1. 实现所有具体方法（100%共性）
2. 实现模板方法（收敛迭代框架）
3. 定义抽象方法接口
4. 提供钩子方法默认实现

### 阶段2：迁移 TimePDESolver

1. 继承 NonlinearPDESolver
2. 实现抽象方法（工厂方法）
3. 保留自己的 solve() 实现（不使用迭代框架）
4. 可选使用 `_train_neural_network_step()`

### 阶段3：迁移 GeneralPDESolver

1. 继承 NonlinearPDESolver
2. 实现抽象方法
3. 可以选择：
   - 保留现有 solve() 实现
   - 或改用 `_convergence_iteration_loop()` 简化代码
4. 覆盖 `_should_train_network()` 返回 True

### 阶段4：实现 FixedPointSolver

1. 继承 NonlinearPDESolver
2. 实现抽象方法
3. **充分利用** `_convergence_iteration_loop()` 框架
4. 覆盖 `_should_train_network()` 返回 False（或配置驱动）

---

## ✅ 总结

### 关键设计决策

1. **保持分离**：三个独立的 Solver 类
2. **提取共性**：基类提供 100% 共性 + 66% 框架
3. **灵活选择**：子类可以选择使用或不使用框架
4. **体现数学**：收敛迭代框架反映了 GeneralPDE 和 FixedPoint 的本质联系

### 共性vs独立性的平衡

```
        共性（基类提供）              独立性（子类自由）
              ↓                           ↓
    [100% 具体方法]              [完全覆盖任何方法]
    [66% 模板方法]    ←平衡→     [选择性使用框架]
    [抽象接口定义]               [自己的 solve()]
              ↓                           ↓
        代码复用 60-70%          概念清晰，扩展灵活
```

这个设计：
- ✅ 充分提取了三者的共性
- ✅ 保持了各自的独立性和灵活性
- ✅ 反映了数学本质（伪时间步 vs 不动点的联系）
- ✅ 代码复用率 60-70%，维护成本大幅降低
