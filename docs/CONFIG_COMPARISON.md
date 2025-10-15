# Time PDE vs General PDE 配置文件对比

## 核心区别

### time_pde: 物理时间演化
```json
{
    "problem_type": "time_pde",
    "spatial_vars": ["x", "y"],           // 空间变量
    "Initial_conditions": [...],          // 给定初始条件
    "time_scheme": "imex_1st",
    "T": 10,                              // 物理时间终止时间
    "dt": 0.1,                            // 物理时间步长（固定）
    "spotter_skip": 1                     // 神经网络训练频率
}
```

### general_pde: 伪时间步至稳态（用户修改版）
```json
{
    "problem_type": "general_pde",
    "spatial_vars": ["x", "y"],           // 空间变量（t 也是空间变量！）
    "Initial_conditions": [...],          // 仅用于兼容性，实际不使用
    "pseudo_time": {                      // 🔥 嵌套的伪时间步参数
        "time_scheme": "imex_1st",
        "max_iterations": 1000,           // 最大迭代次数（替代固定终止时间）
        "convergence_tol": 1e-4,          // 收敛判据
        "dt_init": 0.1,                   // 初始伪时间步
        "dt_min": 1e-5,                   // 最小伪时间步
        "dt_max": 0.5                     // 最大伪时间步
    },
    "spotter_skip": 1
}
```

## 关键差异点

| 字段 | time_pde | general_pde | 说明 |
|------|----------|-------------|------|
| **时间参数组织** | 顶层 `T`, `dt` | 嵌套 `pseudo_time` | ✅ 用户的设计更清晰 |
| **终止条件** | `T` (固定时间) | `max_iterations` + `convergence_tol` | 收敛判据 |
| **时间步** | `dt` (固定或轻微自适应) | `dt_init`, `dt_min`, `dt_max` (自适应) | 伪时间步范围 |
| **初始条件** | `Initial_conditions` (必需) | `Initial_conditions` (兼容性，实际用 Spotter) | general_pde 不使用 |
| **spatial_vars** | `["x", "y"]` | `["x", "y"]` (t 是空间维度) | 注意语义不同 |

## 配置文件设计原则

### ✅ 用户的设计优势：

1. **清晰的语义分离**：
   - time_pde: `T`, `dt` 顶层字段 → 物理时间
   - general_pde: `pseudo_time` 嵌套字段 → 伪时间

2. **避免混淆**：
   - 不会误认为 `dt` 是物理时间步
   - 嵌套结构明确表示"这是伪时间步相关参数"

3. **易于扩展**：
   - 所有伪时间步参数集中在一个字段
   - 未来添加参数无需修改顶层结构

### 修改建议：

我将修改 `GeneralPDEConfig._load_pseudo_time_config()` 以优先读取嵌套的 `pseudo_time` 字段。

## 推荐的配置文件模板

### time_pde 模板
```json
{
    "problem_type": "time_pde",
    "eq": {...},
    "vars_list": ["u"],
    "spatial_vars": ["x", "y"],
    "Initial_conditions": [{...}],
    "boundary_conditions": [...],

    "time_scheme": "imex_1st",
    "T": 10,
    "dt": 0.1,
    "spotter_skip": 1,

    "hidden_dims": [12, 32, 32, 10],
    "epochs_adam": 20000,
    "DNNtol": 0.01,
    ...
}
```

### general_pde 模板
```json
{
    "problem_type": "general_pde",
    "eq": {...},
    "vars_list": ["u"],
    "spatial_vars": ["x", "t"],  // 注意：t 是空间变量
    "Initial_conditions": [{...}],  // 兼容性，实际不使用
    "boundary_conditions": [...],

    "pseudo_time": {
        "time_scheme": "implicit_euler",
        "max_iterations": 1000,
        "convergence_tol": 1e-4,
        "residual_tol": 1e-6,
        "dt_init": 0.1,
        "dt_min": 1e-5,
        "dt_max": 0.5,
        "adaptive_dt": true,
        "dt_increase_factor": 1.2,
        "dt_decrease_factor": 0.5,
        "check_interval": 10,
        "print_interval": 50
    },

    "spotter_skip": 1,

    "hidden_dims": [90, 90, 90, 50],
    "epochs_adam": 10000,
    "DNNtol": 0.0001,
    ...
}
```

## 读取逻辑设计

### GeneralPDEConfig 应该：

1. **优先读取嵌套字段**：
   ```python
   # 优先从 pseudo_time 嵌套字段读取
   pts_params = config_dict.get("pseudo_time", {})

   # 回退到 pseudo_time_stepping 字段（向后兼容）
   if not pts_params:
       pts_params = config_dict.get("pseudo_time_stepping", {})

   # 最后回退到顶层字段（最低优先级）
   self.max_iterations = pts_params.get("max_iterations",
                                        config_dict.get("max_iterations", 1000))
   ```

2. **清晰的日志输出**：
   ```python
   if config_dict.get("pseudo_time"):
       print("✓ Using nested 'pseudo_time' configuration")
   elif config_dict.get("pseudo_time_stepping"):
       print("✓ Using 'pseudo_time_stepping' configuration (legacy)")
   else:
       print("⚠ Using default pseudo-time parameters")
   ```

3. **验证冲突**：
   ```python
   # 如果同时存在 T/dt 和 pseudo_time，发出警告
   if "T" in config_dict or "dt" in config_dict:
       print("⚠ Warning: T/dt found in general_pde config (ignored)")
       print("  Use 'pseudo_time' block for pseudo-time stepping")
   ```
