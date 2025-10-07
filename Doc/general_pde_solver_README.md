# General PDE Solver - 快速入门

**版本:** v1.0
**状态:** ✅ 已完成并测试

---

## 简介

`general_pde_solver` 使用**伪时间步方法**求解任意稳态PDE问题：

```
稳态PDE: L1(U) + L2(U)·F(U) + N(U) = S

伪时间步迭代: ∂U/∂τ + R(U) = 0,  R(U) = L1 + L2·F + N - S
```

**两阶段求解:**
1. **Spotter阶段**: 神经网络学习全局特征 → U_init
2. **Sniper阶段**: 伪时间步迭代 → U_steady

---

## 快速开始

### 1. 准备配置文件

`cases/general_pde_cases/my_problem/config.json`:

```json
{
  "problem_type": "general_pde",
  "problem_name": "My Steady-State PDE",

  "eq": {
    "L1": ["-diff(u,x,2)"],      // ✅ 只有空间导数
    "L2": ["0"],
    "F": ["0"],
    "N": ["u**3 - u"],
    "S": ["0"]
  },

  "spatial_vars": ["x"],          // ✅ 不含t
  "x_domain": [[-1, 1]],
  "n_segments": [3],
  "poly_degree": [10],

  "boundary_conditions": [
    {
      "type": "periodic",
      "pairs": ["left", "right"],
      "var": "u"
    }
  ],

  "initial_condition": {
    "method": "spotter",
    "spotter_config": {
      "epochs": 5000,
      "learning_rate": 0.001
    }
  },

  "pseudo_time_stepping": {
    "max_iterations": 5000,
    "convergence_tol": 1e-6,
    "dt_init": 0.01
  }
}
```

### 2. 运行求解

```python
from src.problem_solvers.general_pde_solver.solver import GeneralPDESolver

# 创建求解器
solver = GeneralPDESolver(case_dir="cases/general_pde_cases/my_problem")

# 求解
results = solver.solve()

# 检查结果
if results["convergence_info"]["converged"]:
    print(f"✓ 收敛! 迭代: {results['convergence_info']['iterations']}")
    U_steady = results["solution"]
else:
    print(f"✗ 未收敛: {results['convergence_info']['reason']}")
```

### 3. 分析结果

```python
import numpy as np

# 解的统计
U = results["solution"]
print(f"||U|| = {np.linalg.norm(U):.6e}")
print(f"范围: [{U.min():.6f}, {U.max():.6f}]")

# 计算PDE残差
residual = solver.fitter.compute_residual(U)
print(f"||R(U)|| = {np.linalg.norm(residual):.6e}")
```

---

## 配置要点

### ⚠️ 关键约束

| 项目 | 正确 ✅ | 错误 ❌ |
|------|--------|---------|
| **spatial_vars** | `["x"]` | `["x", "t"]` |
| **eq.L1** | `"-diff(u,x,2)"` | `"diff(u,t) - diff(u,x,2)"` |
| **problem_type** | `"general_pde"` | `"time_pde"` |

### 推荐参数

```json
{
  "spotter_config": {
    "epochs": 5000,           // 1000-20000
    "learning_rate": 0.001    // 0.0001-0.01
  },
  "pseudo_time_stepping": {
    "max_iterations": 5000,   // 1000-10000
    "convergence_tol": 1e-6,  // 1e-4 ~ 1e-8
    "dt_init": 0.01,          // 0.001 ~ 0.1
    "adaptive_dt": true
  }
}
```

---

## 示例: Allen-Cahn方程

**PDE:**
```
-ε·∂²u/∂x² + 5u³ - 5u = 0,  x ∈ [-1, 1]
周期边界: u(-1) = u(1)
ε = 0.0001
```

**配置:** `cases/general_pde_cases/AC_equation/config.json`

**运行:**
```bash
python3 test_general_pde_ac.py
```

**结果:**
```
Spotter阶段:
  Epoch 1000/1000 | Physics Loss: 1.89e-07
  Initial solution: ||U|| = 8.63e-03  ✅

伪时间步:
  Iter   1 | Residual=4.46e-04 | ||U||=9.06e-03
  Iter  50 | Residual=4.64e-05 | ||U||=2.29e-02
  ...
  ✓ Converged at iteration XXX
    Final residual: < 1e-4  ✅
```

---

## 常见问题

### Q: 出现 `IndexError: index 1 is out of bounds`

**原因:** 配置中 `spatial_vars` 包含了 `"t"`

**解决:**
```json
// ❌ 错误
"spatial_vars": ["x", "t"]

// ✅ 正确
"spatial_vars": ["x"]
```

### Q: 解全是零

**原因:** `_spotter_stage()` 返回值未正确传递 (已在v1.0修复)

**检查:** 更新到最新版本

### Q: 伪时间步不收敛

**解决方案:**
1. 增加Spotter训练轮数: `epochs: 10000`
2. 减小初始时间步: `dt_init: 0.001`
3. 启用自适应: `adaptive_dt: true`

### Q: 源项S不生效

**检查清单:**
- [ ] `config.eq['S']` 已定义
- [ ] 表达式正确 (如 `"sin(pi*x)"`)
- [ ] 使用了 `GeneralPDEDataGenerator`

---

## 文档

- **实施方案:** `Doc/general_pde_solver_implementation_plan.md`
- **最终报告:** `Doc/general_pde_solver_final_report.md`
- **理论背景:** `Doc/伪时间步.md`

---

## 支持的功能

- ✅ 1D/2D/3D稳态PDE
- ✅ 非线性项 (N(U))
- ✅ 源项 (S(x))
- ✅ 周期边界条件
- ✅ 伪时间步迭代
- ✅ 自适应时间步
- ✅ 残差计算
- ✅ 实时可视化

---

**更新时间:** 2025-10-03
**维护者:** DeePoly Team
