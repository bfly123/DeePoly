# 网络迁移功能实现总结

## 背景

**问题：** 原始实现中，`SpotterNet` 独立实现了网络结构（4×128 with weight normalization），无法与 `GeneralPDENet` 共享参数。

**目标：** 使 Spotter 阶段训练的网络可以直接迁移到 Fitter 阶段，实现代码复用。

## 关键概念澄清

对于 `general_pde` 求解器（例如 Allen-Cahn 方程）：

- **物理时间 `t`**：作为**空间维度**之一（例如 (x,t) 是二维空间问题）
- **伪时间 `τ`**：数值迭代的虚拟时间，用于求解稳态解

### Allen-Cahn 方程示例

- **空间问题**：(x,t) ∈ [-1,1]×[0,1]，求解 u(x,t)
- **方程**：∂u/∂t - 0.0001·∂²u/∂x² - 5u + 5u³ = 0
- **伪时间步**：用 τ 迭代求稳态解（通过 `pseudo_time_stepping` 配置控制）

## 实现方案

### 1. 修改 SpotterNet 继承结构

**修改前：**
```python
class SpotterNet(BoundaryConditionMixin, nn.Module):
    def __init__(self, config, layers=[2, 128, 128, 128, 128, 1]):
        # 自定义网络结构...
```

**修改后：**
```python
class SpotterNet(BoundaryConditionMixin, BaseNet):
    def __init__(self, config, use_periodic: bool = False):
        super().__init__(config, use_periodic=use_periodic)
```

**优势：**
- 复用 `BaseNet` 的网络结构
- 继承 `BoundaryConditionMixin` 处理边界条件
- 网络架构与 `GeneralPDENet` 完全一致

### 2. 修改 forward 方法

```python
def forward(self, x: torch.Tensor):
    """兼容 BaseNet 的返回格式"""
    h, U = super().forward(x)
    return h, U  # 返回 (隐藏层输出, 预测解)
```

### 3. 修改采样方法

**修改前：** 硬编码 domain 为 x∈[-1,1], t∈[0,1]

**修改后：** 从 config 读取
```python
def _resample_pde_points(self, n_points: int, device) -> torch.Tensor:
    x_domain = self.config.x_domain
    n_dim = self.config.n_dim
    
    coords = []
    for dim_idx in range(n_dim):
        x_min, x_max = x_domain[dim_idx]
        coord = torch.rand(n_points, 1, ...) * (x_max - x_min) + x_min
        coords.append(coord)
    
    return torch.cat(coords, dim=1)
```

### 4. 修改 solver.py 实现网络迁移

```python
def _spotter_stage(self):
    # 1. 创建 SpotterNet（与 GeneralPDENet 相同架构）
    spotter_model = SpotterNet(self.config, use_periodic=False)
    
    # 2. 训练 SpotterNet
    loss_history = spotter_model.train_spotter(...)
    
    # 3. 迁移参数到 GeneralPDENet
    self.model.load_state_dict(spotter_model.state_dict())
    
    # 4. 提取初始解
    with torch.no_grad():
        _, U_pred = self.model(x_train_tensor)
        U_init = U_pred.cpu().numpy()
    
    return U_init
```

### 5. 更新配置文件

**`cases/general_pde_cases/AC_equation/config.json`：**

```json
{
  "spatial_vars": ["x", "t"],  // t 是空间维度
  "x_domain": [[-1, 1], [0, 1]],  // x∈[-1,1], t∈[0,1]
  "n_segments": [3, 3],
  "poly_degree": [10, 10],
  "initial_condition": {
    "spotter_config": {
      "epochs": 1000,
      "learning_rate": 0.01,
      "n_pde_points": 10000,  // 新增：PDE 采样点数
      "print_interval": 200
    }
  }
}
```

## 测试结果

### 简单网络迁移测试

```bash
$ python test_simple_transfer.py
```

**结果：**
```
✓ ALL TESTS PASSED!
  - Network architectures are compatible (1848 parameters)
  - Parameters can be transferred
  - Transferred network produces identical outputs (diff < 1e-12)
```

### 完整求解器测试

**Spotter 阶段：**
```
Spotter阶段训练 (随机采样 + 自适应学习率 + float64)
训练轮数: 1000
PDE残差点数: 10000 (每epoch随机重采样)
初始学习率: 0.01

Epoch   1000/1000 | Loss: 1.972551e-04 | LR: 1.000000e-04

✓ Spotter训练完成! 总用时: 33.03s
  最佳Loss: 1.320359e-05
```

**网络迁移：**
```
Transferring network weights from SpotterNet to GeneralPDENet...
✓ Network weights transferred successfully
Initial solution: ||U|| = 2.770159e-01, max|U| = 4.151612e-03
```

**Fitter 阶段：**
```
Step 3: Pseudo-Time Stepping to Steady State
Iter 1 | τ=0.000000 | dt=1.000000e-02 | Residual=1.581199e-02
...
```

## 关键改进

1. **代码复用**：SpotterNet 复用 BaseNet 网络结构，减少冗余
2. **参数迁移**：Spotter 训练的网络参数可直接迁移到 Fitter
3. **配置灵活**：从 config 读取 domain，支持任意维度
4. **架构统一**：SpotterNet 和 GeneralPDENet 使用相同架构

## 文件修改清单

1. **`src/problem_solvers/general_pde_solver/core/spotter_init.py`**
   - 继承 `BaseNet` 而非自定义网络
   - 修改 `forward` 返回格式
   - 改进 `_resample_pde_points` 从 config 读取 domain

2. **`src/problem_solvers/general_pde_solver/solver.py`**
   - 导入 `SpotterNet`
   - 修改 `_spotter_stage` 实现网络迁移
   - 添加迁移确认信息

3. **`cases/general_pde_cases/AC_equation/config.json`**
   - 修正 `spatial_vars: ["x", "t"]`（t 是空间维度）
   - 修正 `x_domain: [[-1,1], [0,1]]`
   - 添加 `spotter_config.n_pde_points`

4. **`src/problem_solvers/general_pde_solver/utils/visualize.py`**
   - 修复 scatter 调用参数错误

## 后续建议

1. **性能优化**：考虑使用 GPU 加速训练
2. **early stopping**：实现 Spotter 阶段的提前停止
3. **warmup**：考虑从 Spotter 的学习率平滑过渡到 Fitter
4. **checkpoint**：保存中间训练结果，便于调试
