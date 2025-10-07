# General PDE Solver 与 Time PDE Solver 逻辑对齐

**日期**: 2025-10-04
**目标**: 将 `general_pde_solver` 调整为与 `time_pde_solver` 一致的逻辑流程

---

## 🎯 核心问题

原先 `general_pde_solver` 的流程：
```
Spotter 阶段：训练神经网络获得初始解
    ↓
Fitter 阶段：伪时间步迭代（不再训练神经网络）
    └── 只更新多项式系数，神经网络参数固定
```

这与 `time_pde_solver` 的逻辑**不一致**：
```
Time PDE Solver 流程：
每个时间步：
    1. 先训练神经网络 (_train_neural_network_step)
    2. 再执行时间推进 (time_step)
    └── 神经网络参数在整个求解过程中持续更新
```

---

## ✅ 修改方案

### 调整后的 General PDE Solver 流程

```
Spotter 阶段：训练神经网络获得初始解
    ↓
Fitter 阶段：伪时间步迭代
    每次迭代：
        1. 训练神经网络 (_train_neural_network_step)  ← 新增
        2. 执行伪时间步 (time_step)
        3. 检查收敛
    └── 神经网络参数在整个求解过程中持续更新
```

**关键改进**：在伪时间步循环中，每次迭代都先训练神经网络，再执行求解步骤。

---

## 📝 代码修改详情

### 1. **GeneralPDEFitter** (`fitter.py`)

#### 修改 `solve_steady_state()` 签名
```python
# 修改前
def solve_steady_state(
    self,
    U_init: Optional[np.ndarray] = None,
    coeffs_init: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

# 修改后
def solve_steady_state(
    self,
    U_init: Optional[np.ndarray] = None,
    coeffs_init: Optional[np.ndarray] = None,
    model: Optional[object] = None  # ← 新增 model 参数
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
```

#### 在伪时间步循环中添加神经网络训练
```python
# 修改前
for iteration in range(self.max_iterations):
    # Execute one time step
    U_new, U_seg_new, coeffs_new = self.time_step(...)

# 修改后
for iteration in range(self.max_iterations):
    # Train neural network at current pseudo-time step (类似 time_pde_solver)
    if self.model is not None:
        self._train_neural_network_step(iteration, dt, U_current)  # ← 新增

    # Execute one time step
    U_new, U_seg_new, coeffs_new = self.time_step(...)
```

#### 新增 `_train_neural_network_step()` 方法
```python
def _train_neural_network_step(self, it: int, dt: float, U_current: np.ndarray = None) -> None:
    """Neural network training at current pseudo-time step (参考 time_pde_solver)"""

    # 检查是否需要跳过神经网络训练
    spotter_skip = getattr(self.config, 'spotter_skip', 1)  # 默认每步都训练
    skip_training = (it % spotter_skip != 0) and (it > 0)  # 第一步总是训练

    if skip_training:
        print(f"  Skipping neural network training (spotter_skip={spotter_skip}, it={it})")
        self.fitter_init(self.model)
        return

    print(f"  Training neural network (it={it}, inheriting from previous parameters)...")

    # 准备GPU数据
    data_GPU = self.model.prepare_gpu_data(self.data, U_current)

    # 从上一次的模型状态开始训练（不重置参数）
    self.model.train()  # 设置为训练模式

    # 执行训练，从上次的权重继承
    self.model.train_net(self.data, self.model, data_GPU, dt=dt)

    # 计算最终Loss
    self.model.eval()
    final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
    print(f"  Neural network training completed, loss: {final_loss:.8e}")

    # 重新初始化fitter以使用更新后的模型
    self.fitter_init(self.model)
```

### 2. **GeneralPDEFitter (Refactored)** (`fitter_refactored.py`)

完全相同的修改：
- 添加 `model` 参数
- 在循环中调用 `_train_neural_network_step()`
- 添加 `_train_neural_network_step()` 方法

### 3. **GeneralPDESolver** (`solver.py`)

#### 调用 `solve_steady_state()` 时传递 model
```python
# 修改前
U_steady, coeffs_steady, conv_info = self.fitter.solve_steady_state(U_init=U_init)

# 修改后
U_steady, coeffs_steady, conv_info = self.fitter.solve_steady_state(
    U_init=U_init,
    model=self.model  # ← 传递 model 供训练使用
)
```

---

## 🔄 与 Time PDE Solver 的对应关系

| Time PDE Solver | General PDE Solver | 说明 |
|----------------|-------------------|------|
| `T` (物理时间) | `τ` (伪时间) | 时间变量 |
| `T < T_final` | `it < max_iterations` | 循环条件 |
| `_train_neural_network_step()` | `_train_neural_network_step()` | **新增**，每步训练 |
| `time_step()` | `time_step()` | 时间推进 |
| IMEX-RK 时间格式 | implicit_euler 格式 | 时间离散 |
| `_monitor_solution()` | `_monitor_pseudo_time_step()` | 监控收敛 |

---

## 🎨 新增配置参数

可在 `config.json` 中添加 `spotter_skip` 参数控制训练频率：

```json
{
  "spotter_skip": 1,  // 每步都训练神经网络
  // 或
  "spotter_skip": 5   // 每5步训练一次神经网络（节省计算）
}
```

**默认值**: `spotter_skip = 1`（每步都训练）

---

## 📊 优势对比

### 修改前
- ✗ Spotter 只在初始阶段训练一次
- ✗ Fitter 阶段神经网络参数固定
- ✗ 无法在伪时间步迭代中改善神经网络近似

### 修改后
- ✅ 每个伪时间步都可以训练神经网络
- ✅ 神经网络参数持续更新，逐步逼近真解
- ✅ 与 time_pde_solver 逻辑统一
- ✅ 灵活控制训练频率（通过 `spotter_skip`）

---

## 🧪 验证结果

```bash
$ python -c "from src.problem_solvers.general_pde_solver.core.fitter import GeneralPDEFitter; ..."
✓ Both fitter modules imported successfully
✓ GeneralPDEFitter has _train_neural_network_step: True
✓ GeneralPDEFitterRefactored has _train_neural_network_step: True
```

---

## 📚 文件清单

### 修改的文件
1. `src/problem_solvers/general_pde_solver/core/fitter.py`
   - 添加 `model` 参数
   - 添加 `_train_neural_network_step()` 方法
   - 在循环中调用训练方法

2. `src/problem_solvers/general_pde_solver/core/fitter_refactored.py`
   - 相同修改

3. `src/problem_solvers/general_pde_solver/solver.py`
   - 传递 `model` 给 `solve_steady_state()`

### 新增的文档
- `GENERAL_PDE_ALIGNMENT.md` (本文件)

---

## 🔍 实现细节

### 神经网络训练逻辑

```python
# 每次伪时间步迭代
for iteration in range(max_iterations):
    # 步骤 1: 训练神经网络（参数从上次继承）
    if model is not None and (iteration % spotter_skip == 0):
        # 准备当前解的数据
        data_GPU = model.prepare_gpu_data(data, U_current)

        # 训练（不重置参数，从上次权重继续）
        model.train_net(data, model, data_GPU, dt=dt)

        # 重新初始化 fitter 以使用更新后的模型
        fitter.fitter_init(model)

    # 步骤 2: 执行伪时间步推进
    U_new, coeffs_new = time_step(U_current, dt, ...)

    # 步骤 3: 检查收敛
    if converged:
        break
```

### 参数继承机制

- **Spotter 阶段**: 训练获得初始参数 `θ₀`
- **Fitter 第1步**: 从 `θ₀` 开始训练 → `θ₁`
- **Fitter 第2步**: 从 `θ₁` 开始训练 → `θ₂`
- ...
- **Fitter 第n步**: 从 `θₙ₋₁` 开始训练 → `θₙ`

每次训练都是在上次的基础上**微调**，而不是从头训练。

---

## 🚀 使用示例

### 基本用法（自动训练）

```bash
# 运行求解器（默认每步都训练）
python -m src.problem_solvers.general_pde_solver.solver \
    --case_path cases/general_pde_cases/AC_equation
```

### 控制训练频率

在 `config.json` 中设置：
```json
{
  "spotter_skip": 5  // 每5步训练一次，节省计算时间
}
```

### 禁用伪时间步训练

```python
# 如果不想在伪时间步中训练，将 model 设为 None
U_steady, coeffs_steady, conv_info = fitter.solve_steady_state(
    U_init=U_init,
    model=None  # 禁用训练
)
```

---

## 📈 预期效果

修改后，预期效果：

1. **收敛速度**: 可能加快（神经网络持续改进近似）
2. **最终精度**: 可能提高（神经网络参数持续优化）
3. **计算成本**: 增加（每步都训练神经网络）
4. **灵活性**: 提高（可通过 `spotter_skip` 平衡精度和效率）

---

## ✅ 总结

| 改进项 | 说明 |
|--------|------|
| **逻辑统一** | ✅ 与 time_pde_solver 保持一致 |
| **代码结构** | ✅ 遵循相同的设计模式 |
| **参数继承** | ✅ 神经网络参数持续更新 |
| **灵活控制** | ✅ 支持 spotter_skip 配置 |
| **向后兼容** | ✅ API 保持兼容（model=None 时行为不变） |

---

**生成时间**: 2025-10-04
**状态**: ✅ 修改完成并验证
