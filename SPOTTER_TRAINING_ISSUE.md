# SpotterNet 训练问题诊断

## 当前状态

### ✓ 已完成
1. **网络结构复用** - SpotterNet 继承 BaseNet，与 GeneralPDENet 架构一致
2. **参数迁移** - load_state_dict() 成功迁移 (1848 参数，误差 < 1e-12)
3. **PDE 方程修正** - 添加 `diff(u,t)` 项（t 作为空间维度）
4. **config 修正** - spatial_vars: ["x", "t"], x_domain: [[-1,1], [0,1]]

### ✗ 问题
**训练解幅度太小：** `max|U| ≈ 0.007` (期望 O(1) 量级)

## 对比分析

### standalone_pinn_ac_improved.py
```python
# 边界条件 (BC):
u_boundary = -1.0  # 强制 u(±1,t) = -1
x_left/right = ±1, t ∈ [0,1]  # 1000个点

# 初始条件 (IC):
u_init = x² * cos(πx)  # t=0时刻
x ∈ [-1,1]  # 1000个点

# 损失函数:
loss = loss_pde + 10 * loss_bc
loss_bc = ||u_pred(BC) - u_bc||²
```

### SpotterNet (当前)
```python
# 边界条件设置:
{
  "type": "dirichlet",
  "location": "left/right",
  "region": "x",  # ← 2D不支持
  "value": "-1.0"
}

# 问题:
1. 边界条件未正确应用 (Warning: region 'x' not supported for 2D)
2. 初始条件可能未施加
3. BC loss 权重可能不够
```

## 解决方案

### 方案1: 修改 SpotterNet 数据生成
直接在 `SpotterNet.prepare_gpu_data()` 或 `train_spotter()` 中：
1. 手动生成边界条件点 (x=±1, t随机)
2. 手动生成初始条件点 (t=0, x随机)  
3. 在 loss 中明确施加这些约束

### 方案2: 修改 data generator
扩展 GeneralPDEDataGenerator 支持 2D 边界条件和初始条件

### 方案3: 暂时简化问题
先用 standalone 版本验证网络训练效果，确认问题确实是边界/初始条件

##推荐方案

**短期:** 方案1 - 在 SpotterNet 中直接处理 BC/IC
**长期:** 方案2 - 完善 data generator

## 下一步

1. 在 `SpotterNet.train_spotter()` 中添加 BC/IC 数据生成
2. 修改 `physics_loss_steady_state()` 确保 BC loss 权重足够
3. 验证训练后 max|U| 达到 O(1) 量级
