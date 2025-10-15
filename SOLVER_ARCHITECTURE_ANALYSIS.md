# Solver 架构一致性分析与重构建议

## 当前架构对比

### 1. TimePDESolver (时间演化)

**循环位置**: ✅ solver.py:175-225

```python
def solve_time_evolution(self):
    """时间演化主循环 - 在 solver 层"""
    it, T, dt, U, U_seg, coeffs = self._initialize_time_evolution()

    while T < self.config.T:  # 循环在 solver
        dt = self._compute_adaptive_timestep(it, T, dt, U)

        # DNN 训练（solver 层统一管理）
        self._train_neural_network_step(it, dt, U_current=U)

        # 单步执行（fitter 只做数学运算）
        U, U_seg, coeffs = self.fitter.solve_time_step(U, U_seg, dt, ...)

        T += dt
        it += 1
```

**优点**:
- ✅ 循环在 solver 层
- ✅ DNN 训练在 solver 统一管理
- ✅ fitter 只负责单步数学运算
- ✅ 命名简洁直观

**缺点**:
- ⚠️ 使用局部变量管理状态（不利于扩展）

---

### 2. PseudoTimePDESolver (伪时间步迭代)

**循环位置**: ✅ solver.py:143-179

```python
def _run_pseudotime_loop(self, state: Dict[str, Any]):
    """伪时间步主循环 - 在 solver 层"""
    for iteration in range(self.fitter.max_iterations):  # 循环在 solver
        # DNN 训练 + 单步执行
        success, state = self._execute_iteration_step(iteration, state)

        # 收敛检查
        conv_status = self._check_and_update_state(iteration, state)
        if conv_status == 'converged':
            return state['U'], state['coeffs'], conv_info
```

**优点**:
- ✅ 循环在 solver 层
- ✅ DNN 训练在 solver 统一管理
- ✅ 使用 state 字典管理状态（易扩展）

**缺点**:
- ❌ 方法命名冗长：`_run_pseudotime_loop`, `_execute_iteration_step`, `_check_and_update_state`
- ❌ 方法职责不清：`_check_and_update_state` 混合了检查和更新两个职责
- ❌ 与 TimePDESolver 命名不一致

---

### 3. FixIterationPDESolver (固定点迭代)

**循环位置**: ⚠️ 两处都有！

**A. solver.py:59-150** ✅ 正确位置
```python
def _solve_with_dnn_training(self, U_init):
    """固定点迭代主循环 - 在 solver 层"""
    for iteration in range(max_iterations):  # 循环在 solver
        # DNN 训练（solver 层统一管理）
        self._train_neural_network_step(iteration, dt_virtual, U_current)

        # 单步执行（fitter 只做数学运算）
        coeffs_new, U_new = self.fitter.fit(U_current, coeffs_current)

        if residual < convergence_tol: return ...
```

**B. fitter.py:42-133** ❌ 错误位置（冗余）
```python
def solve_steady_state(self, U_init):
    """固定点迭代主循环 - 在 fitter 层（冗余！）"""
    for iteration in range(self.max_iterations):  # 循环在 fitter（错误！）
        coeffs_new = super().fit(U_current=U_current)
        U_new, _ = self.construct(...)

        if residual < self.convergence_tol: return ...
```

**问题**:
- ❌ **架构冲突**: solver.py 已有循环，fitter.py 又重复实现
- ❌ **职责混乱**: fitter 不应管理迭代循环，只应实现单步
- ❌ **DNN 训练**: fitter 中的循环无法使用 solver 的统一 DNN 训练

**解决方案**: 删除 `FixIterationPDEFitter.solve_steady_state`

---

## 架构统一原则

### ✅ 正确的职责分离

| 层级 | 职责 | 示例方法 |
|------|------|---------|
| **Solver** | 迭代循环、DNN训练、收敛监控、可视化 | `solve()`, `_solve_*()`, `_train_neural_network_step()` |
| **Fitter** | 单步数学运算、算子构建 | `time_step()`, `_build_segment_jacobian()` |
| **TimeScheme** | 时间格式细节 | `build_stage_jacobian()` |

### ✅ 统一的命名规范

#### 1. 主求解入口（用户调用）
```python
def solve(self, U_init=None) -> Dict[str, Any]:
    """所有 solver 统一入口"""
```

#### 2. 内部循环方法（solver 层）

| Solver 类型 | 建议命名 | 当前命名 | 状态 |
|------------|---------|---------|------|
| TimePDE | `solve_time_evolution()` | `solve_time_evolution()` | ✅ 保持 |
| PseudoTimePDE | `solve_steady_state()` | `_run_pseudotime_loop()` | ❌ 需改 |
| FixIteration | `solve_steady_state()` | `_solve_with_dnn_training()` | ⚠️ 可改 |

**统一为**:
- 时间演化: `solve_time_evolution()`
- 稳态问题: `solve_steady_state()`

#### 3. 单步执行方法（fitter 层）

| Solver 类型 | 建议命名 | 当前命名 | 状态 |
|------------|---------|---------|------|
| TimePDE | `time_step()` | `solve_time_step()` | ✅ 接近 |
| PseudoTimePDE | `time_step()` | `time_step()` | ✅ 正确 |
| FixIteration | `fixed_point_step()` | ❌ 无（用的fit） | ❌ 需加 |

#### 4. 辅助方法命名

| 功能 | 推荐命名 | PseudoTime当前 | 改进 |
|------|---------|---------------|------|
| 初始化状态 | `_initialize_iteration()` | `_initialize_iteration_state()` | ✅ 接近 |
| 单次迭代 | `_perform_iteration_step()` | `_execute_iteration_step()` | ⚠️ 可简化 |
| 更新状态 | `_update_iteration_state()` | `_check_and_update_state()` | ❌ 职责混合 |
| 检查收敛 | `_check_convergence()` | 混在 `_check_and_update_state()` | ❌ 需拆分 |
| 打印进度 | `_print_iteration_progress()` | `_print_iteration_progress()` | ✅ 正确 |

---

## 重构建议

### 优先级 1: 删除 FixIterationPDEFitter 冗余循环

**文件**: `src/problem_solvers/nonlinear_pde_solver/fixiteration_pde_solver/core/fitter.py`

**操作**:
1. ❌ 删除 `solve_steady_state()` 方法（第42-133行）
2. ✅ 保留 `_build_segment_jacobian()` 方法
3. ✅ solver.py 中的 `_solve_with_dnn_training()` 已经正确实现循环

**影响**: 无，solver.py 已有完整实现

---

### 优先级 2: 简化 PseudoTimePDESolver 命名

**文件**: `src/problem_solvers/nonlinear_pde_solver/pseudotime_pde_solver/solver.py`

**当前**:
```python
def solve(self):
    state = self._initialize_iteration_state(U_init)
    U_final, coeffs_final, conv_info = self._run_pseudotime_loop(state)

def _run_pseudotime_loop(self, state):
    success, state = self._execute_iteration_step(iteration, state)
    conv_status = self._check_and_update_state(iteration, state)
```

**建议改为**:
```python
def solve(self):
    state = self._initialize_iteration(U_init)
    U_final, coeffs_final, conv_info = self.solve_steady_state(state)

def solve_steady_state(self, state):
    """统一命名：稳态问题求解"""
    success, state = self._perform_iteration_step(iteration, state)

    # 拆分职责
    if self._check_convergence(state):
        return ...
    self._update_iteration_state(iteration, state)
```

**改进**:
- ✅ 与 FixIteration 命名统一（都叫 `solve_steady_state`）
- ✅ 方法命名更简洁（`_perform` 替代 `_execute`）
- ✅ 职责分离（拆分 check 和 update）

---

### 优先级 3: 统一状态管理（可选）

**建议**: 使用 `IterationState` 数据类（已定义在 `solver.py:21`）

**好处**:
- 类型安全
- IDE 自动补全
- 统一接口

**改造示例**:
```python
# 当前（字典）
state = {
    'U': U_init,
    'coeffs': coeffs_init,
    'dt': dt,
    'residual': None
}

# 建议（数据类）
from ..solver import IterationState
state = IterationState(
    U=U_init,
    coeffs=coeffs_init,
    dt=dt,
    custom={'residual': None}
)
```

---

## 实施计划

### 第一阶段（高优先级）
1. ✅ 删除 `FixIterationPDEFitter.solve_steady_state()`
2. ✅ 重命名 `PseudoTimePDESolver._run_pseudotime_loop()` → `solve_steady_state()`
3. ✅ 拆分 `_check_and_update_state()` 为 `_check_convergence()` + `_update_iteration_state()`

### 第二阶段（中优先级）
4. ⚠️ 统一命名：`_execute_iteration_step()` → `_perform_iteration_step()`
5. ⚠️ FixIteration 重命名：`_solve_with_dnn_training()` → `solve_steady_state()`（可选）

### 第三阶段（低优先级）
6. 🔄 引入 `IterationState` 替代字典（可选，影响较大）

---

## 架构对比表（重构后）

| 特性 | TimePDE | PseudoTimePDE | FixIteration |
|------|---------|---------------|--------------|
| **循环位置** | solver.py | solver.py | solver.py |
| **主方法** | `solve_time_evolution()` | `solve_steady_state()` | `solve_steady_state()` |
| **单步方法** | `time_step()` | `time_step()` | `fit()` 或新增 `fixed_point_step()` |
| **DNN 训练** | solver 层 | solver 层 | solver 层 |
| **状态管理** | 局部变量 | state 字典 | 局部变量 |
| **命名风格** | 简洁 | ✅ 简化后简洁 | 简洁 |

---

## 总结

### 核心问题
1. ❌ **FixIteration**: fitter 层有冗余循环（需删除）
2. ❌ **PseudoTime**: 方法命名冗长且职责混合（需简化）
3. ⚠️ **命名不统一**: 稳态求解器应统一叫 `solve_steady_state()`

### 核心原则
- ✅ **循环在 solver**: 所有迭代循环都在 solver 层
- ✅ **单步在 fitter**: fitter 只实现单步数学运算
- ✅ **统一命名**: 同类功能使用相同方法名

### 预期收益
- 代码更易理解和维护
- 新增求解器更容易
- DNN 训练逻辑统一
- 架构清晰，职责分明
