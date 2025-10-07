# General PDE Solver 重构计划

## 目标
参考 `time_pde_solver` 的结构，重构 `general_pde_solver`，使其更加模块化和统一。

## 当前问题
1. `GeneralPDEFitter` 继承自 `TimePDEFitter`，但实现了独立的 `solve_steady_state()` 方法
2. 伪时间步逻辑直接写在 fitter 中，不够模块化
3. 没有利用 `time_schemes` 框架

## 核心区别

### time_pde_solver (时间演化)
```python
# 主循环
while T < T_final:
    dt = adaptive_dt(...)
    U, U_seg, coeffs = fitter.solve_time_step(U, U_seg, dt, coeffs)
    T += dt
```

### general_pde_solver (伪时间步到稳态)
```python
# 主循环
while residual > tol and iter < max_iter:
    dt_tau = adaptive_dt_tau(...)
    U, U_seg, coeffs = fitter.solve_time_step(U, U_seg, dt_tau, coeffs)
    tau += dt_tau
    residual = compute_residual(U)
```

## 关键洞察

**两者本质上是相同的！**
- 都是时间推进：`∂U/∂t + F(U) = 0` 或 `∂U/∂τ + R(U) = 0`
- 都使用隐式格式求解
- 都需要自适应时间步
- 都调用 `fitter.solve_time_step()`

**唯一区别**:
1. **循环终止条件**:
   - time_pde: `T < T_final` (达到最终时间)
   - general_pde: `residual < tol` (达到稳态)

2. **时间格式**:
   - time_pde: 可选 IMEX-RK(2,2,2), IMEX-1st 等
   - general_pde: 固定 implicit_euler (一阶隐式)

## 重构方案

### 方案1: 最小改动 (推荐)

**保持当前架构，只优化 GeneralPDEFitter**

```python
class GeneralPDEFitter(TimePDEFitter):
    """使用伪时间步的稳态PDE求解器"""

    def solve_steady_state(self, U_init, coeffs_init=None):
        """主求解循环 - 参考 TimePDESolver.solve_time_evolution()"""

        # 初始化
        it, tau, dt_tau, U, U_seg, coeffs = self._initialize_pseudo_time(U_init, coeffs_init)

        # 主循环
        while tau < max_tau:
            # 计算残差
            residual = self._compute_residual(U, coeffs)

            # 检查收敛
            if residual < self.residual_tol:
                print(f"✓ Converged! Residual = {residual:.6e}")
                break

            # 检查停滞
            if self._check_stagnation():
                print(f"✗ Stagnation detected")
                break

            # 自适应时间步
            dt_tau = self._compute_adaptive_dt(residual, dt_tau)

            # 求解一步 (复用 TimePDEFitter.solve_time_step)
            U, U_seg, coeffs = self.solve_time_step(
                U, U_seg, dt_tau, coeffs_n=coeffs, step=it
            )

            # 更新
            tau += dt_tau
            it += 1

            # 记录历史
            self._record_iteration(it, tau, dt_tau, residual, U)

            # 打印进度
            if it % self.print_interval == 0:
                print(f"Iter {it:5d} | τ={tau:.6f} | dt={dt_tau:.6e} | Residual={residual:.6e}")

        return U, coeffs, self._get_convergence_info()
```

**优点**:
- 最小改动
- 复用 `TimePDEFitter.solve_time_step()` 和 time_schemes
- 代码结构与 `TimePDESolver.solve_time_evolution()` 对齐

**改动文件**:
- `src/problem_solvers/general_pde_solver/core/fitter.py` - 重构 `solve_steady_state()`

### 方案2: 完全统一 (更彻底但改动大)

**创建统一的时间推进框架**

```python
class BaseTimeMarchingSolver:
    """统一的时间推进基类"""

    def solve(self):
        """通用时间推进循环"""
        it, time_var, dt, U, U_seg, coeffs = self._initialize()

        while not self._termination_condition(it, time_var, U):
            dt = self._adaptive_dt(...)
            U, U_seg, coeffs = self.fitter.solve_time_step(...)
            time_var += dt
            it += 1
            self._monitor(it, time_var, U)

        return U, coeffs

# 子类实现
class TimePDESolver(BaseTimeMarchingSolver):
    def _termination_condition(self, it, T, U):
        return T >= self.config.T_final

class GeneralPDESolver(BaseTimeMarchingSolver):
    def _termination_condition(self, it, tau, U):
        residual = self._compute_residual(U)
        return residual < self.tol or it >= self.max_iter
```

**优点**:
- 完全统一的架构
- 代码复用最大化

**缺点**:
- 改动太大
- 可能影响现有功能

## 推荐实施步骤

### Step 1: 重构 GeneralPDEFitter.solve_steady_state()

参考 `TimePDESolver.solve_time_evolution()` 的结构：

```python
# 原来的实现（在fitter内部）
def solve_steady_state(...):
    # 初始化
    # 大循环
    # 收敛判断

# 新实现（参考time_pde结构）
def solve_steady_state(...):
    # 1. 初始化
    it, tau, dt, U, U_seg, coeffs = self._initialize_pseudo_time(...)

    # 2. 主循环
    while tau < max_tau:
        # 2.1 计算当前残差
        residual = self._compute_residual_norm(U, coeffs)

        # 2.2 检查收敛
        if self._check_convergence(residual, it):
            break

        # 2.3 自适应时间步
        dt = self._compute_adaptive_pseudo_dt(residual, dt, it)

        # 2.4 求解一步 (复用父类方法!)
        U, U_seg, coeffs = self.solve_time_step(
            U, U_seg, dt, coeffs_n=coeffs, step=it
        )

        # 2.5 更新
        tau += dt
        it += 1

        # 2.6 监控和可视化
        self._monitor_pseudo_time_step(it, tau, dt, residual, U)

    return U, coeffs, convergence_info
```

### Step 2: 提取辅助方法

将当前 `solve_steady_state()` 中的逻辑拆分成小方法：

```python
def _initialize_pseudo_time(self, U_init, coeffs_init):
    """初始化伪时间步"""
    ...

def _compute_residual_norm(self, U, coeffs):
    """计算PDE残差范数"""
    ...

def _check_convergence(self, residual, it):
    """检查是否收敛"""
    ...

def _compute_adaptive_pseudo_dt(self, residual, dt_current, it):
    """自适应伪时间步"""
    ...

def _monitor_pseudo_time_step(self, it, tau, dt, residual, U):
    """监控并打印进度"""
    ...
```

### Step 3: 统一输出格式

使两个solver的输出保持一致：

```python
# time_pde_solver
time_history = [...]
solution_history = [...]

# general_pde_solver (对应)
pseudo_time_history = [...]  # tau values
solution_history = [...]      # U values at each iteration
```

### Step 4: 复用可视化

统一使用 `visualizer.update_animation_data()`:

```python
# time_pde中
self.visualizer.update_animation_data(it, T, ...)

# general_pde中（类似）
self.visualizer.update_animation_data(it, tau, ...)
```

## 时间格式选择

对于 general_pde，固定使用 `implicit_euler`:

```python
# 在 GeneralPDESolver.__init__() 或 GeneralPDEFitter.__init__()
self.fitter = GeneralPDEFitter(
    config=config,
    data=data_train,
    time_scheme="implicit_euler"  # 固定为一阶隐式
)
```

## 预期效果

重构后：

1. ✅ **代码结构统一**: general_pde 和 time_pde 使用相同的模式
2. ✅ **复用time_schemes**: 伪时间步也使用 `implicit_euler` scheme
3. ✅ **更清晰的逻辑**: 主循环在 fitter.solve_steady_state() 中，与 time_evolution() 对应
4. ✅ **更好的可维护性**: 模块化的辅助方法
5. ✅ **一致的监控和可视化**: 复用 visualizer

## 关键代码对比

### time_pde_solver.solver.solve_time_evolution()
```python
while T < self.config.T:
    dt = self._compute_adaptive_timestep(...)
    U, U_seg, coeffs = self.fitter.solve_time_step(U, U_seg, dt, coeffs)
    T += dt
    self.visualizer.update_animation_data(...)
```

### general_pde_solver.fitter.solve_steady_state() (重构后)
```python
while not converged:
    residual = self._compute_residual_norm(U, coeffs)
    if residual < tol: break

    dt_tau = self._compute_adaptive_pseudo_dt(residual, dt_tau)
    U, U_seg, coeffs = self.solve_time_step(U, U_seg, dt_tau, coeffs)
    tau += dt_tau
    self.visualizer.update_animation_data(...)  # 如果有的话
```

## 实施优先级

1. **高优先级** (立即实施):
   - 重构 `GeneralPDEFitter.solve_steady_state()` 的主循环结构
   - 提取辅助方法 (`_compute_residual_norm`, `_check_convergence` 等)

2. **中优先级** (后续优化):
   - 统一可视化接口
   - 改进自适应时间步策略

3. **低优先级** (长期目标):
   - 考虑创建统一的 `BaseTimeMarchingSolver`

---

**结论**: 采用**方案1（最小改动）**，重点重构 `GeneralPDEFitter.solve_steady_state()` 方法，使其结构与 `TimePDESolver.solve_time_evolution()` 对齐，同时保持功能不变。
