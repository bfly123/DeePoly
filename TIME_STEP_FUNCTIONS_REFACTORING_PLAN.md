# 时间步相关函数重构方案

## 问题描述

当前在 `BaseDeepPolyFitter` 中有以下与时间步相关的函数：

```python
# base_fitter.py:822-873
def solve_time_step(...)  # 执行一步时间步
def get_time_scheme_info(...)  # 获取时间格式信息
def validate_operators(...)  # 验证算子配置
def estimate_stable_dt(...)  # 估计稳定时间步
def print_time_scheme_summary(...)  # 打印时间格式摘要
```

**核心问题**：这些函数放在 Fitter 中不合适，因为：
1. 三个非线性 solver 对时间步的处理方式完全不同
2. 这些函数更多是"控制逻辑"而非"算子编译/执行"

## 三个 Solver 的时间步处理差异

### 1. TimePDESolver（物理时间演化）
- **时间含义**：真实物理时间 t
- **时间格式**：IMEX-RK(2,2,2), OneStep Predictor等
- **自适应策略**：基于CFL条件、稳定性分析、解变化率
- **终止条件**：T >= T_final
- **需要的函数**：
  - `solve_time_step()` ✅ 执行物理时间推进
  - `get_time_scheme_info()` ✅ 获取IMEX格式信息
  - `estimate_stable_dt()` ✅ CFL稳定性估计
  - `validate_operators()` ✅ 验证L1/L2/N/F配置

**示例代码（solver.py:249-290）**：
```python
def _compute_adaptive_timestep(self, it, T, dt, U):
    base_dt = self.config.dt
    dt = base_dt

    # CFL条件限制
    if hasattr(self.config, 'cfl_number'):
        cfl_dt = self._compute_cfl_timestep(U)
        dt = min(dt, cfl_dt)

    # 稳定性限制
    if hasattr(self.fitter, "estimate_stable_dt"):
        dt_stable = self.fitter.estimate_stable_dt(U)  # ❌ 应该在solver中
        dt = min(dt, dt_stable)
```

### 2. PseudoTimePDESolver（伪时间迭代）
- **时间含义**：伪时间 τ（迭代参数）
- **时间格式**：Implicit Euler
- **自适应策略**：基于残差变化率（fitter中实现）
- **终止条件**：残差 < tol 或达到最大迭代次数
- **需要的函数**：
  - `solve_time_step()` ❌ 只需要 `time_step()`，不需要包装
  - `get_time_scheme_info()` ⚠️ 简化版（只有Implicit Euler）
  - `estimate_stable_dt()` ❌ 不需要（基于残差调整）
  - `validate_operators()` ✅ 验证算子配置

**示例代码（pseudotime solver.py:254-257）**：
```python
# 自适应时间步由fitter管理
state['dt'], state['stagnation_count'] = self.fitter._update_adaptive_timestep(
    residual, state['residual_prev'], state['stagnation_count']
)
```

### 3. FixIterationPDESolver（固定点迭代）
- **时间含义**：无（dt_virtual 仅用于神经网络训练）
- **时间格式**：无
- **自适应策略**：无
- **终止条件**：残差 < tol 或达到最大迭代次数
- **需要的函数**：
  - `solve_time_step()` ❌ 不需要
  - `get_time_scheme_info()` ❌ 不需要
  - `estimate_stable_dt()` ❌ 不需要
  - `validate_operators()` ✅ 验证算子配置

**示例代码（fixiteration solver.py:94-101）**：
```python
# 固定点迭代，无需时间步管理
for iteration in range(max_iterations):
    dt_virtual = 0.01  # 仅用于NN训练
    self._train_neural_network_step(iteration, dt_virtual, U_current)
    coeffs_new, U_new = self.fitter.fit(U_current, coeffs_current)
```

## 职责分离原则分析

### Fitter 的核心职责
✅ **应该包含**：
- 算子编译和缓存（`_precompile_all_operators`）
- 约束处理（`_add_constraints`）
- 特征生成（`_get_features`）
- **单步算子执行**（`time_step` - 核心算子调用）
- 雅可比矩阵构建（`_build_jacobian`）
- 全局/段级转换（`global_to_segments`）

❌ **不应该包含**：
- 时间步控制策略（自适应、CFL等）
- 时间格式信息查询（solver层决策）
- 迭代循环管理
- 可视化和报告

### Solver 的核心职责
✅ **应该包含**：
- 迭代循环管理
- 自适应时间步控制
- 神经网络训练调度
- **时间格式选择和参数设置**
- 收敛判断和监控
- 可视化和后处理

## 重构方案

### 方案 A：分层职责重构（推荐）

#### 1. BaseFitter 保留的内容
```python
class BaseDeepPolyFitter:
    """基础拟合器 - 只负责算子编译和执行"""

    def __init__(self, config, dt=None, data=None, time_scheme=None):
        self.time_scheme = None  # 保留，因为time_step需要它
        if time_scheme:
            self._initialize_time_scheme(time_scheme)

    def time_step(self, U_n, U_seg, dt, coeffs_n=None, **kwargs):
        """执行单步 - 核心算子调用（保留）"""
        if self.time_scheme is None:
            raise RuntimeError("Time scheme not initialized")
        return self.time_scheme.time_step(U_n, U_seg, dt, coeffs_n, **kwargs)

    # ❌ 移除以下函数：
    # - solve_time_step (包装器，移到solver)
    # - get_time_scheme_info (查询接口，移到solver)
    # - validate_operators (验证逻辑，移到solver)
    # - estimate_stable_dt (控制逻辑，移到solver)
    # - print_time_scheme_summary (UI逻辑，移到solver)
```

#### 2. NonlinearPDESolver 基类新增通用方法
```python
class NonlinearPDESolver(ABC):
    """非线性求解器基类 - 管理时间步控制"""

    def validate_operator_configuration(self) -> Dict[str, bool]:
        """验证算子配置（通用）"""
        if not hasattr(self.fitter, 'time_scheme') or self.fitter.time_scheme is None:
            return {"valid": False, "error": "No time scheme"}

        return self.fitter.time_scheme.validate_operators()

    def get_solver_time_info(self) -> Dict[str, Any]:
        """获取求解器时间信息（通用接口，子类覆盖）"""
        return {
            "solver_type": self.__class__.__name__,
            "has_time_scheme": hasattr(self.fitter, 'time_scheme')
        }
```

#### 3. TimePDESolver 特定方法
```python
class TimePDESolver(NonlinearPDESolver):
    """物理时间演化求解器"""

    def get_solver_time_info(self) -> Dict[str, Any]:
        """获取时间格式信息"""
        info = super().get_solver_time_info()
        if self.fitter.time_scheme:
            info.update(self.fitter.time_scheme.get_scheme_info())
        return info

    def estimate_stable_dt(self, U_current: np.ndarray, safety_factor: float = 0.8) -> float:
        """估计稳定时间步（仅TimePDE需要）"""
        if not hasattr(self.fitter, 'time_scheme') or self.fitter.time_scheme is None:
            raise RuntimeError("No time scheme initialized")
        return self.fitter.time_scheme.estimate_stable_dt(U_current, safety_factor)

    def print_time_scheme_summary(self):
        """打印时间格式摘要"""
        info = self.get_solver_time_info()
        print("=== Time Integration Scheme ===")
        print(f"Scheme: {info.get('scheme_name', 'N/A')}")
        print(f"Order: {info.get('order', 'N/A')}")
        print(f"Type: {info.get('scheme_type', 'N/A')}")
        print("=" * 30)

    def _compute_adaptive_timestep(self, it, T, dt, U):
        """计算自适应时间步（TimePDE特定）"""
        base_dt = self.config.dt

        # CFL条件
        if hasattr(self.config, 'cfl_number'):
            cfl_dt = self._compute_cfl_timestep(U)
            dt = min(base_dt, cfl_dt)

        # 稳定性限制（现在是solver方法）
        if hasattr(self.config, 'use_stability_limit'):
            dt_stable = self.estimate_stable_dt(U)  # ✅ solver方法
            dt = min(dt, dt_stable)

        # 确保不超过最终时间
        if T + dt > self.config.T:
            dt = self.config.T - T

        return dt
```

#### 4. PseudoTimePDESolver 特定方法
```python
class PseudoTimePDESolver(NonlinearPDESolver):
    """伪时间迭代求解器"""

    def get_solver_time_info(self) -> Dict[str, Any]:
        """获取伪时间信息"""
        info = super().get_solver_time_info()
        info.update({
            "time_type": "pseudo_time",
            "scheme": "implicit_euler",
            "adaptive": True,
            "residual_based": True
        })
        return info

    # ✅ 自适应策略由fitter管理（基于残差）
    # 不需要estimate_stable_dt和CFL计算
```

#### 5. FixIterationPDESolver 特定方法
```python
class FixIterationPDESolver(NonlinearPDESolver):
    """固定点迭代求解器"""

    def get_solver_time_info(self) -> Dict[str, Any]:
        """获取迭代信息"""
        info = super().get_solver_time_info()
        info.update({
            "solver_type": "fixed_point_iteration",
            "has_time_stepping": False,
            "convergence_based": True
        })
        return info

    # ✅ 不需要任何时间步相关方法
```

### 方案对比

| 函数 | 当前位置 | 方案A | 理由 |
|-----|---------|-------|------|
| `time_step` | BaseFitter | BaseFitter | 核心算子执行，必须在fitter |
| `solve_time_step` | BaseFitter | ❌ 删除 | 无用包装器 |
| `get_time_scheme_info` | BaseFitter | TimePDESolver | 仅TimePDE需要查询 |
| `validate_operators` | BaseFitter | NonlinearPDESolver | 通用验证逻辑 |
| `estimate_stable_dt` | BaseFitter | TimePDESolver | 仅TimePDE需要 |
| `print_time_scheme_summary` | BaseFitter | TimePDESolver | UI逻辑在solver |

## 向后兼容性迁移

### 阶段1：添加新接口（不破坏现有代码）
```python
# base_fitter.py
def solve_time_step(self, ...):
    """Deprecated: Use time_step() directly or solver.solve_time_step()"""
    import warnings
    warnings.warn("solve_time_step is deprecated, use time_step()", DeprecationWarning)
    return self.time_step(...)

def get_time_scheme_info(self):
    """Deprecated: Use solver.get_solver_time_info()"""
    import warnings
    warnings.warn("get_time_scheme_info moved to solver", DeprecationWarning)
    if self.time_scheme:
        return self.time_scheme.get_scheme_info()
    return {}
```

### 阶段2：更新调用点
```python
# TimePDESolver.solve_time_evolution() - BEFORE
U, U_seg, coeffs = self.fitter.solve_time_step(U, U_seg, dt, ...)

# TimePDESolver.solve_time_evolution() - AFTER
U, U_seg, coeffs = self.fitter.time_step(U, U_seg, dt, ...)
```

### 阶段3：移除废弃方法（主版本更新）

## 实施步骤

1. ✅ **验证当前调用点**
   - 搜索所有 `solve_time_step` 的调用
   - 搜索所有 `get_time_scheme_info` 的调用
   - 搜索所有 `estimate_stable_dt` 的调用

2. ✅ **在NonlinearPDESolver中添加新方法**
   - `validate_operator_configuration()`
   - `get_solver_time_info()` (抽象)

3. ✅ **在TimePDESolver中实现特定方法**
   - `get_solver_time_info()`
   - `estimate_stable_dt()`
   - `print_time_scheme_summary()`
   - 更新 `_compute_adaptive_timestep()` 使用solver方法

4. ✅ **在PseudoTimePDESolver中实现**
   - `get_solver_time_info()`

5. ✅ **在FixIterationPDESolver中实现**
   - `get_solver_time_info()`

6. ✅ **更新BaseFitter**
   - 添加废弃警告
   - 保留 `time_step()` 核心方法

7. ✅ **测试和验证**
   - 运行所有三个solver的测试用例
   - 确保无回归

8. ✅ **文档更新**

## 总结

这个重构方案：
1. ✅ **明确职责**：Fitter负责算子执行，Solver负责控制策略
2. ✅ **符合差异**：三个solver的时间步处理各不相同
3. ✅ **向后兼容**：通过废弃警告平滑过渡
4. ✅ **可扩展性**：未来新增solver只需实现自己的时间控制
5. ✅ **代码清晰**：减少了Fitter中不必要的接口

**核心原则**：
- Fitter = 算子执行层（what to compute）
- Solver = 控制决策层（how to iterate）
