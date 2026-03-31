# Pseudo-Time 配置收敛方案

## 1. 目标

当前 `pseudo_time` 配置同时混入了 4 类信息：

- 用户真正需要调的伪时间推进参数
- Spotter 网络训练参数
- 求解器内部固定实现细节
- 调试/保底类收敛参数

这会导致 case `config.json` 过长，也容易让后续新增算例把无效参数继续复制下去。

本次收敛目标：

- 将 Spotter 训练参数从 `pseudo_time` 和顶层主配置中拆出，统一放入 `spotter`
- 将伪时间求解器内部固定行为改成默认后台策略，不再要求 case 显式填写
- 保留旧配置兼容，避免已有算例和测试失效

## 2. 新的推荐配置面

### 2.1 `pseudo_time`

推荐仅保留下面这组用户常调参数：

- `max_iterations`
- `residual_tol`
- `dt_init`
- `dt_min`
- `dt_max`
- `adaptive_dt`
- `check_interval`
- `print_interval`
- `convergence_primary_var`
- `convergence_primary_tol`

如果是多方程稳态问题，并且需要参考旧 `2D_NS_Stable` 的质量项处理，还保留下面这组“算法语义参数”：

- `pseudo_time_formulation`
- `mass_eq_mask`
- `mass_var_map`
- `frozen_nonlinear_jacobian`

### 2.2 `spotter`

Spotter 训练参数统一写入：

```json
"spotter": {
  "train_every": 5,
  "epochs_adam": 200,
  "epochs_lbfgs": 100,
  "max_retries": 1,
  "tol": 0.03
}
```

字段含义：

- `train_every`: 每隔多少个伪时间步重新训练一次 Spotter，内部映射到旧字段 `spotter_skip`
- `epochs_adam`: Adam 训练轮数
- `epochs_lbfgs`: LBFGS 训练轮数
- `max_retries`: 训练失败后的重试次数
- `tol`: 训练损失阈值，内部映射到旧字段 `DNNtol`

## 3. 后台固定或默认化的参数

下面这些参数对当前 `PseudoTimePDESolver` 不建议继续在 case 中显式维护：

- `time_scheme`
  当前伪时间实现固定为 `implicit_euler`，即使配置里写其他值，也不会切换真实后端。
- `save_interval`
  当前伪时间迭代主循环没有消费该参数，继续保留只会制造“配置有效”的错觉。
- `steady_state_mode`
  当前 pseudo-time 求解器默认按稳态模式工作，case 不需要重复声明。
- `dt_increase_factor`
- `dt_decrease_factor`
- `convergence_improvement_factor`
- `convergence_plateau_steps`
- `convergence_plateau_tol`
- `convergence_dt_growth_tol`
- `rollback_on_plateau`
- `rollback_dt_min`
- `force_retrain_on_plateau`
- `return_best_on_exit`

上面这组参数不是全部删除，而是优先走代码默认值。只有在做收敛故障排查或控制策略实验时，才建议重新暴露到 case。

## 4. 兼容策略

为避免老算例失效，当前加载逻辑保留以下兼容路径：

- 旧顶层字段 `epochs_adam`
- 旧顶层字段 `epochs_lbfgs`
- 旧顶层字段 `max_retries`
- 旧顶层字段 `DNNtol`
- 旧顶层字段 `spotter_skip`
- 旧块 `fitter_training`

优先级约定：

- `spotter` 最高
- `fitter_training` 次之
- 顶层旧字段最低

这保证了老 case 不需要一次性全部重写，但新 case 应统一使用 `spotter`。

## 5. 已应用到的算例

本次已经将下面两个 pseudo-time case 调整到新的推荐写法：

- `cases/nonlinear_pde_cases/pseudotime_pde_cases/lid_driven_cavity_re100/config.json`
- `cases/nonlinear_pde_cases/pseudotime_pde_cases/New_test_2D/config.json`

## 6. 严格回归测试要求

每次继续清理 pseudo-time 配置时，至少执行下面几组回归：

- 配置加载单测
  `pytest -q test/test_pseudotime_config.py`
- 多方程方腔流配置与初始化
  `pytest -q test/test_lid_driven_cavity_case.py`
- 伪时间相关公共回归
  `pytest -q test/test_pseudotime_visualizer_helpers.py test/test_pseudotime_fitter_helpers.py`
- 多方程与 CLI 回归
  `pytest -q test/test_multi_equation_support.py test/test_func_fitting_multi_equation_case.py test/test_cli.py`

如果这次收敛影响到可视化、固定点求解或数据装配，还需要补跑：

- `pytest -q test/test_dataset_visualization_mixin.py`
- `pytest -q test/test_fixiteration_fitter.py test/test_fixiteration_config.py test/test_fixiteration_visualizer.py`
- `pytest -q test/test_base_data_helpers.py test/test_shape_helpers.py`

## 7. 下一步建议

完成本轮配置收敛后，下一步建议继续两件事：

- 将 plateau/restart 控制参数封装成内部 controller policy，避免继续占用 case 配置面
- 为 pseudo-time 增加一份独立的 schema 文档或导出模板，避免新 case 再次复制旧字段
