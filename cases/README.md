# Cases Directory Guide

`cases/` 是项目的案例资产根目录，当前同时承载三类内容：

- 可执行案例输入
- 参考数据
- 运行结果产物

现阶段 solver 和测试对 case 目录结构有硬依赖，因此不应直接迁移已有 case 路径。整理策略应以“约定明确、资产分层、结果隔离”为主，而不是直接重排目录。

## 分类

- `cases/func_fitting_cases/`
  - 函数拟合案例
  - 标准输入通常包含 `config.json`、`data_generate.py`、`output.py`
  - `multi_equation_2d_poly/` 可作为多方程 2D `func_fitting` 的最小回归基准，覆盖 `u/v` 双变量、分段拟合和结果输出
- `cases/linear_pde_cases/`
  - 线性 PDE 案例
  - 标准输入通常包含 `config.json`、`data_generate.py`、`output.py`
- `cases/nonlinear_pde_cases/`
  - 非线性 PDE 案例
  - 主要分为 `fixiteration_pde_cases/`、`pseudotime_pde_cases/`、`Time_pde_cases/`
  - 标准输入通常包含 `config.json`，部分案例还包含 `reference_data/`

## 推荐标准布局

每个案例目录应尽量保持以下结构：

```text
case_name/
  config.json
  data_generate.py        # 仅函数拟合/线性 PDE 常见
  output.py               # 仅函数拟合/线性 PDE 常见
  reference_data/         # 非线性 PDE 参考数据
  results/                # 运行产物
```

## 目录边界

- `config.json`
  - 必须视为运行入口的一部分
- `data_generate.py`
  - 视为案例级数据准备逻辑
- `output.py`
  - 视为案例级后处理或可视化逻辑
- `reference_data/`
  - 视为静态输入资产，不应与运行结果混放
- `results/`
  - 仅用于运行输出，不应再承载配置、源码和参考输入

## 当前问题

- 多个 `results/` 目录中存在大体积产物，拉低仓库级架构指标
- `cases/*/output.py` 为案例逻辑的一部分，但命名上容易与运行产物混淆
- `func_fitting_cases/case_2d_large_gradient/` 当前为空目录，需要补说明或移除
- 非线性案例中的 `reference_data/*.m` 和 `.mat` 文件属于参考资产，应与结果目录严格隔离

## 整理规则

- 新增案例时，优先复用上述标准布局
- 不要把模型、图像、误差表和中间 `.npy/.npz/.dat` 文件放回案例根目录
- 所有运行输出都写入 `results/`
- 所有静态参考输入都写入 `reference_data/`
- 非必要情况下，不要让测试直接依赖 `results/` 中的历史产物

## 后续治理建议

- 为 `cases/` 建立统一的结果产物保留策略
- 将说明性文档收敛到每个大类目录的 `README.md`
- 逐步减少 repo 中已跟踪的大型结果文件，只保留最小 smoke 所需样例
