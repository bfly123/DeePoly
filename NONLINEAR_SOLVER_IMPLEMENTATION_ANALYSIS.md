# NonlinearPDESolver 实现分析与 FixedPointSolver 设计

**日期**: 2025-10-10
**目标**: 分析三种Solver的共性和差异，设计FixedPointSolver，提取共性算法到基类

---

## 📊 三种Solver的对比分析

### 1. TimePDESolver - 物理时间演化

```python
特征：
├─ Fitter: TimePDEFitter
├─ 核心方法: time_step(U, U_seg, dt, coeffs, ...)
├─ 迭代变量: 物理时间 T
├─ 终止条件: T >= T_final （时间驱动）
├─ 时间步: dt （固定或自适应）
├─ 训练策略: 每隔 spotter_skip 步训练一次
└─ 输出: 完整时间演化历史 + 最终解
```

**数学形式**:
```
(U^{n+1} - U^n)/dt + L1(U^{n+1}) + L2(U^{n+1})·F(U^{n+1}) + N(U^{n+1}) = S
```

### 2. GeneralPDESolver - 伪时间步至稳态

```python
特征：
├─ Fitter: GeneralPDEFitter (继承 TimePDEFitter)
├─ 核心方法: time_step(U, U_seg, dτ, coeffs, ...)  # 伪时间步
├─ 迭代变量: 伪时间 τ
├─ 终止条件: residual < tol 或 iter >= max_iter （收敛驱动）
├─ 时间步: dτ （自适应，根据残差调整）
├─ 训练策略: 每隔 spotter_skip 步训练（默认每步训练）
├─ 初始化: Spotter网络预训练
└─ 输出: 稳态解 + 收敛信息
```

**数学形式**:
```
(U^{n+1} - U^n)/dτ + L1(U^{n+1}) + L2(U^{n+1})·F(U^{n+1}) + N(U^{n+1}) = S
```
当 τ → ∞，U^{n+1} → U^n，左边第一项趋于0，达到稳态。

**自适应策略**:
```python
if residual > 0.1:     dτ = dt_max          # 大残差→快速推进
elif residual > 0.01:  dτ = dt_max * 0.5    # 中等残差
elif residual > 1e-4:  dτ = dt_max * 0.2    # 小残差
else:                  dτ = dt_min          # 接近收敛→精确逼近
```

### 3. FixedPointSolver - 不动点迭代 （需要实现）

```python
特征：
├─ Fitter: FixedPointFitter （新建，建议继承 BaseDeepPolyFitter）
├─ 核心方法: fit(U_current) → U_new  # 固定N(U^n)，求解线性系统
├─ 迭代变量: 迭代次数 iter
├─ 终止条件: residual < tol 或 iter >= max_iter （收敛驱动）
├─ 时间步: 无时间步概念
├─ 训练策略: 通常不训练（或仅初始化时训练一次）
├─ 初始化: Spotter网络预训练 或 zero/given
└─ 输出: 稳态解 + 收敛信息
```

**数学形式**:
```
L1(U^{n+1}) + L2(U^{n+1})·F(U^{n+1}) + N(U^n) = S
                                         ↑
                                    固定为上一步的值
```

**关键特点**:
- 没有 `(U^{n+1} - U^n)/dτ` 项
- 非线性项 N 使用 U^n（上一步值），而不是 U^{n+1}
- 每步求解一个线性系统（N(U^n)当做已知项）
- 相当于 GeneralPDE 当 dτ → ∞ 的极限情况

---

## 🔍 核心差异分析

| 维度 | TimePDE | GeneralPDE | FixedPoint |
|------|---------|------------|------------|
| **Fitter基类** | TimePDEFitter | TimePDEFitter | **BaseDeepPolyFitter** |
| **核心方法** | `time_step()` | `time_step()` | **`fit()`** |
| **时间项** | `(U^{n+1}-U^n)/dt` | `(U^{n+1}-U^n)/dτ` | **无** |
| **非线性项** | `N(U^{n+1})` | `N(U^{n+1})` | **`N(U^n)`** |
| **终止条件** | 时间达到 | 残差收敛 | 残差收敛 |
| **自适应** | dt可选 | dτ自适应 | 无 |
| **训练频率** | 每隔k步 | 每步（可选） | 不训练或初始 |
| **Spotter** | 不需要 | 需要 | 可选 |

---

## 💡 共性算法提取

### 1. 初始化算法 ✅ 可抽象到基类

三种solver都需要初始猜测值，策略相同：

```python
# 已在 NonlinearPDESolver 中实现
def _get_initial_guess(self) -> np.ndarray:
    """默认实现：零初值"""
    n_points = self.data_train['x'].shape[0]
    n_eqs = self.config.n_eqs
    return np.zeros((n_points, n_eqs))

# 需要添加到基类
def _initialize_from_config(self) -> np.ndarray:
    """根据配置的initial_condition生成初值

    支持三种方法：
    1. 'zero': 零初值
    2. 'spotter': 神经网络预训练
    3. 'given': 从Initial_conditions表达式计算
    """
    init_method = getattr(self.config, 'initial_condition', {}).get('method', 'zero')

    if init_method == 'zero':
        return self._get_initial_guess()
    elif init_method == 'spotter':
        return self._spotter_initialization()
    elif init_method == 'given':
        return self._config_based_initialization()
    else:
        raise ValueError(f"Unknown initial_condition method: {init_method}")

def _spotter_initialization(self) -> np.ndarray:
    """使用Spotter网络预训练初始化

    通用流程：
    1. 创建SpotterNet（使用与主网络相同的架构）
    2. 训练Spotter（最小化物理损失）
    3. 转移权重到主网络
    4. 提取预测值作为初值
    """
    # 通用实现，可被GeneralPDE和FixedPoint共用
    pass

def _config_based_initialization(self) -> np.ndarray:
    """根据Initial_conditions表达式计算初值

    从config.Initial_conditions读取，如：
    Initial_conditions:
      - var: u
        value: "sin(2*pi*x)"
      - var: v
        value: "cos(2*pi*x)"
    """
    # 通用实现
    pass
```

### 2. 收敛监控算法 ✅ 已在基类

```python
# 已实现
def _compute_residual(self, U_new, U_old) -> float
def _update_convergence_state(self, U_old, U_new, state, conv_tol) -> IterationState
```

GeneralPDE 和 FixedPoint 都需要这些方法。

### 3. 神经网络训练管理 ✅ 已在基类

```python
# 已实现
def _train_neural_network_step(self, iter, dt, U_current, force=False)
def _should_train_network(self, iter) -> bool
```

- TimePDE: 每隔 spotter_skip 步训练
- GeneralPDE: 每隔 spotter_skip 步训练（默认1，即每步）
- FixedPoint: 覆盖 `_should_train_network()` 返回 False（不训练）

### 4. 后处理算法 ⚠️ 部分可抽象

```python
# 已实现
def _interpolate_to_test_grid(self, U_train) -> np.ndarray

# 可添加到基类（通用框架）
def _postprocess_results(self, **kwargs):
    """后处理框架（子类覆盖具体实现）"""
    # 1. 插值到test grid
    # 2. 可视化
    # 3. 保存结果
    # 4. 与参考解对比（如果有）
```

### 5. 自适应步长算法 ⚠️ GeneralPDE特有（但可提供接口）

```python
# 在基类提供默认实现（不自适应）
def _adapt_pseudo_timestep(self, dt: float, residual: float) -> float:
    """自适应伪时间步调整 - 钩子方法

    默认实现：不调整
    GeneralPDE 覆盖：根据残差分段调整
    """
    return dt
```

---

## 🏗️ FixedPointFitter 设计

### 继承关系建议

```python
BaseDeepPolyFitter
    ├── TimePDEFitter (提供 time_step)
    │   └── GeneralPDEFitter (复用 time_step 作为伪时间步)
    └── FixedPointFitter (提供 fit，不需要 time_step)
```

**原因**:
- FixedPoint 从概念上不应该有"时间步"
- 继承 BaseDeepPolyFitter 可以复用：
  - Operators预编译
  - Features生成
  - 分段处理
  - 基础线性代数操作
- 不继承 TimePDEFitter 避免不必要的时间步相关代码

### FixedPointFitter 核心方法

```python
class FixedPointFitter(BaseDeepPolyFitter):
    """不动点迭代Fitter - 固定N(U^n)求解线性系统"""

    def fit(
        self,
        U_current: np.ndarray,
        coeffs_prev: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """固定N(U^n)，求解线性系统得到U^{n+1}

        求解: L1(U^{n+1}) + L2(U^{n+1})·F(U^{n+1}) = S - N(U^n)

        Args:
            U_current: 当前解 U^n
            coeffs_prev: 上一步系数（用于初始化，可选）

        Returns:
            U_new: 新解 U^{n+1}
            coeffs_new: 新系数
        """
        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        # 1. 转换为分段格式
        U_seg_current = self.global_to_segments(U_current)

        # 2. 对每个分段求解
        coeffs_new = []
        U_seg_new = []

        for seg_idx in range(self.ns):
            U_seg_i = U_seg_current[seg_idx]
            features = self._features[seg_idx][0]

            # 计算固定的非线性项 N(U^n)
            N_fixed = self.N_func(features, U_seg_i)

            # 获取源项 S
            S = self._get_source_term(seg_idx)

            # 构造有效源项: S_eff = S - N(U^n)
            S_eff = S - N_fixed

            # 获取算子矩阵
            L1_ops = self._linear_operators[seg_idx].get("L1", None)
            L2_ops = self._linear_operators[seg_idx].get("L2", None)

            # 求解线性系统（类似time_step，但无时间导数项）
            coeffs_seg = self._solve_linear_system_segment(
                L1_ops, L2_ops, features, S_eff, U_seg_i, coeffs_prev
            )

            coeffs_new.append(coeffs_seg)

            # 构造新解
            U_seg_new_i = self._construct_solution_from_coeffs(
                coeffs_seg, features, U_seg_i
            )
            U_seg_new.append(U_seg_new_i)

        # 3. 转换回全局格式
        coeffs_new = np.array(coeffs_new)
        U_new, _ = self.construct(self.data, self.model, coeffs_new)

        return U_new, coeffs_new

    def _solve_linear_system_segment(
        self,
        L1_ops: List,
        L2_ops: List,
        features: np.ndarray,
        S_eff: np.ndarray,  # 有效源项 = S - N(U^n)
        U_seg: np.ndarray,
        coeffs_init: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """求解单个分段的线性系统

        最小化: ||L1(c) + L2(c)·F(U) - S_eff||^2

        注意：这里的F(U)使用当前U_seg计算，但不是未知量
        在迭代过程中，U_seg是固定的（来自U^n）

        这是一个关于系数c的线性最小二乘问题
        """
        # 实现线性求解器
        # 可以使用：
        # 1. 直接法（QR分解、SVD等）- 适合小规模
        # 2. 迭代法（GMRES、BiCGSTAB等）- 适合大规模
        pass

    def solve_steady_state(
        self,
        U_init: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        conv_tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """不动点迭代至稳态

        迭代格式: U^{n+1} = FixedPointMap(U^n)
        其中 FixedPointMap 由 fit() 方法定义
        """
        # 初始化
        if U_init is None:
            U_init = np.zeros((len(self.data["x"]), self.config.n_eqs))

        U_current = U_init.copy()
        coeffs_current = None
        residual_history = []

        print("=" * 70)
        print("Fixed-Point Iteration")
        print("=" * 70)
        print(f"Max iterations: {max_iterations}")
        print(f"Convergence tolerance: {conv_tol:.2e}")
        print("=" * 70)

        for iteration in range(max_iterations):
            # 执行一步不动点迭代
            U_new, coeffs_new = self.fit(U_current, coeffs_current)

            # 计算残差
            residual = np.linalg.norm(U_new - U_current) / (np.linalg.norm(U_new) + 1e-10)
            residual_history.append(residual)

            # 打印进度
            if iteration % 10 == 0 or residual < conv_tol:
                print(f"Iter {iteration+1:4d} | Residual: {residual:.6e} | ||U||: {np.linalg.norm(U_new):.6e}")

            # 检查收敛
            if residual < conv_tol:
                print("\n" + "=" * 70)
                print(f"✓ Converged at iteration {iteration+1}!")
                print(f"  Final residual: {residual:.6e}")
                print("=" * 70)

                conv_info = {
                    "converged": True,
                    "iterations": iteration + 1,
                    "final_residual": residual,
                    "residual_history": residual_history
                }
                return U_new, coeffs_new, conv_info

            # 更新
            U_current = U_new.copy()
            coeffs_current = coeffs_new.copy()

        # 未收敛
        print(f"\n✗ Maximum iterations ({max_iterations}) reached")
        conv_info = {
            "converged": False,
            "reason": "max_iterations",
            "iterations": max_iterations,
            "final_residual": residual,
            "residual_history": residual_history
        }
        return U_current, coeffs_current, conv_info
```

---

## 🎯 FixedPointSolver 实现

```python
class FixedPointSolver(NonlinearPDESolver):
    """不动点迭代求解器"""

    def solve(self, U_init: Optional[np.ndarray] = None) -> Dict:
        """使用统一迭代框架实现不动点迭代"""
        print("\n" + "="*70)
        print("Fixed-Point Iteration Solver")
        print("="*70)

        # Step 1: 初始化
        if U_init is None:
            U_init = self._initialize_from_config()  # 使用基类的通用初始化

        self.fitter.fitter_init(self.model)

        # Step 2: 配置参数
        max_iter = getattr(self.config, 'max_iterations', 100)
        conv_tol = getattr(self.config, 'convergence_tol', 1e-6)

        # Step 3: 初始状态
        initial_state = IterationState(
            iter=0,
            residual=float('inf'),
            converged=False,
            custom={'max_iter': max_iter, 'conv_tol': conv_tol}
        )

        # Step 4: 定义迭代步骤
        def iteration_step(U: np.ndarray, state: IterationState):
            """不动点迭代步骤"""
            # 固定N(U^n)，求解线性系统
            coeffs = self.fitter.fit(U_current=U)
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

        # Step 5: 定义终止条件
        def termination_check(state: IterationState) -> bool:
            return state.converged or state.iter >= state.custom['max_iter']

        # Step 6: 定义进度打印
        def print_progress(state: IterationState):
            if state.iter % 10 == 0 or state.residual < conv_tol:
                print(f"Iteration {state.iter}: residual = {state.residual:.6e}")

        # Step 7: 使用统一迭代框架
        U_final, final_state = self._unified_iteration_loop(
            U_init=U_init,
            initial_state=initial_state,
            iteration_step=iteration_step,
            termination_check=termination_check,
            print_progress=print_progress
        )

        # Step 8: 后处理
        conv_info = {
            "converged": final_state.converged,
            "iterations": final_state.iter,
            "final_residual": final_state.residual
        }

        self._postprocess_results(U_final, final_state.coeffs, conv_info)

        return {
            "solution": U_final,
            "coefficients": final_state.coeffs,
            "convergence_info": conv_info
        }

    # 工厂方法实现
    def _create_config(self, case_dir: str):
        from src.problem_solvers.fixed_point_solver.utils.config import FixedPointConfig
        return FixedPointConfig(case_dir=case_dir)

    def _create_fitter(self, time_scheme: Optional[str]):
        from src.problem_solvers.fixed_point_solver.core.fitter import FixedPointFitter
        return FixedPointFitter(config=self.config, data=self.data_train)

    def _get_default_time_scheme(self) -> Optional[str]:
        return None  # FixedPoint 不需要时间格式

    def _should_train_network(self, iter: int) -> bool:
        """FixedPoint 通常不训练网络"""
        return False  # 或者只在iter==0时返回True（初始训练）
```

---

## 📋 需要添加到 NonlinearPDESolver 的通用方法

### 1. 初始化方法增强

```python
def _initialize_from_config(self) -> np.ndarray:
    """根据配置生成初值（通用）"""

def _spotter_initialization(self) -> np.ndarray:
    """Spotter网络初始化（通用）"""

def _config_based_initialization(self) -> np.ndarray:
    """表达式初始化（通用）"""
```

### 2. 自适应步长接口（钩子方法）

```python
def _adapt_pseudo_timestep(self, dt: float, residual: float) -> float:
    """自适应步长调整 - 默认不调整，GeneralPDE覆盖"""
    return dt
```

---

## 🚀 实现步骤建议

### Phase 1: 增强基类
1. 在 NonlinearPDESolver 添加初始化方法
   - `_initialize_from_config()`
   - `_spotter_initialization()`
   - `_config_based_initialization()`

2. 添加自适应步长钩子
   - `_adapt_pseudo_timestep()`

### Phase 2: 实现 FixedPointFitter
1. 创建 `src/problem_solvers/fixed_point_solver/core/fitter.py`
2. 继承 BaseDeepPolyFitter
3. 实现 `fit()` 方法
4. 实现 `solve_steady_state()` 方法

### Phase 3: 实现 FixedPointSolver
1. 创建 `src/problem_solvers/fixed_point_solver/solver.py`
2. 继承 NonlinearPDESolver
3. 实现 `solve()` 使用统一迭代框架
4. 实现工厂方法

### Phase 4: 配置和工具类
1. FixedPointConfig
2. FixedPointDataGenerator (可能复用GeneralPDEDataGenerator)
3. FixedPointNet (可能复用GeneralPDENet)
4. FixedPointVisualizer

---

## 🎨 三种Solver的统一迭代框架应用对比

| 组件 | TimePDE | GeneralPDE | FixedPoint |
|------|---------|------------|------------|
| **iteration_step** | `time_step(U, dt)` | `time_step(U, dτ)` + train | `fit(U)` |
| **termination_check** | `T >= T_final` | `converged or iter >= max` | `converged or iter >= max` |
| **state_update** | dt固定或自适应 | dτ自适应（分段） | 无 |
| **print_progress** | 打印T, dt | 打印τ, dτ, residual | 打印iter, residual |

**代码复用率**:
- 基类提供：~450行（统一框架 + 工具）
- TimePDESolver：~100行
- GeneralPDESolver：~120行（+ 自适应逻辑）
- FixedPointSolver：~80行（最简单）
- **总复用率：~80%**

---

## 📝 总结

1. **共性算法已抽象**: 初始化、收敛判断、神经网络训练、后处理
2. **FixedPointFitter设计**: 继承BaseDeepPolyFitter，提供fit()方法
3. **统一迭代框架**: 三种solver都通过回调函数参数化差异
4. **需要补充**: 初始化方法增强（spotter、config-based）

**关键洞察**:
- 不动点迭代是伪时间步当 dτ → ∞ 的极限情况
- 三种solver本质上都是迭代，只是迭代步骤和终止条件不同
- 通过统一迭代框架 + 工厂方法模式，实现了高达80%的代码复用
