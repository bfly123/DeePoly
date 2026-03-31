# DeePoly 架构优化回归测试清单

## 1. 文档目的

本文档定义架构优化过程中的严格回归测试门禁，用于确保重构不会破坏已有功能、案例运行、自动代码生成和可视化能力。

配套主文档：

- 总体方案：[`docs/architecture-optimization-plan.md`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/docs/architecture-optimization-plan.md)
- 执行计划：[`docs/architecture-optimization-execution-plan.md`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/docs/architecture-optimization-execution-plan.md)

## 2. 回归测试总原则

- 每个阶段都必须执行与其改动范围对应的最小完整回归
- 改动核心抽象层时，必须补单元测试和组件测试
- 改动入口层或代码生成层时，必须补集成测试
- 改动性能关键路径时，必须附 benchmark
- 无 `cupy` 环境必须作为一类正式测试环境

## 3. 环境矩阵

建议至少覆盖以下环境：

### 环境 A：CPU-only

- Python 可运行
- 无 `cupy`
- 用于验证 import、可视化、基础 solver、代码生成

### 环境 B：GPU-available

- `cupy` 可用
- 用于验证 GPU 相关 solver/backend

若当前阶段没有 GPU 环境，至少保证：

- CPU-only 环境全部测试通过
- GPU 路径具备清晰降级逻辑

## 3.1 当前已执行的严格回归命令（2026-03-19）

本轮架构优化后，已实际执行并通过的回归包括：

- `python -m compileall src/abstract_class/config test/test_base_data_helpers.py`
- `python -m compileall src/meta_coding test/test_auto_snipper.py`
- `python -m compileall src/problem_solvers/nonlinear_pde_solver/fixiteration_pde_solver/utils test/test_fixiteration_config.py`
- `pytest -q test/test_base_data_helpers.py`
- `pytest -q test/test_config_base_helpers.py test/test_base_data_helpers.py`
- `pytest -q test/test_auto_snipper.py test/test_auto_eq.py test/test_equation_processor.py test/test_operator_factory.py test/test_operator_usage_examples.py test/test_config_base_helpers.py test/test_base_data_helpers.py`
- `pytest -q test/test_fixiteration_config.py test/test_import_smoke.py test/test_config_base_helpers.py test/test_pseudotime_config.py`
- `pytest -q test/test_integration_with_solver.py`
- `python -m compileall src/problem_solvers/nonlinear_pde_solver/time_pde_solver/utils test/test_time_config.py`
- `python -m compileall src/meta_coding test/test_auto_replace_nonlinear.py`
- `pytest -q test/test_time_config.py test/test_time_pde_visualizer.py test/test_import_smoke.py test/test_integration_with_solver.py`
- `pytest -q test/test_auto_replace_nonlinear.py test/test_auto_eq.py test/test_auto_snipper.py test/test_equation_processor.py test/test_operator_factory.py test/test_import_smoke.py`
- `pytest -q test/test_time_pde_visualizer.py test/test_new_test_2d_visualization.py test/test_dimensional_visualizer.py test/test_imex_first_order.py test/test_time_config.py test/test_import_smoke.py test/test_integration_with_solver.py`

说明：

- 集成测试中暴露的 `visualization` dict/object 兼容问题已被修复并重新通过
- 运行中保留的 warning 主要来自无 `cupy` 环境、Matplotlib `Axes3D` 和 PyTorch NumPy bridge，当前视为环境型 warning，不视为功能回归失败
- 最新复测显示：`src` 核心代码已达到 `9.0/10`，仓库级总分仍受 `cases/` 和文档/示例文件影响

## 4. 全局回归分层

## 4.1 导入层回归

必须覆盖：

- `import src`
- `import src.visualization`
- `import src.problem_solvers`
- `import src.meta_coding`
- `from src.algebraic_solver.linear_solver import LinearSolver`

通过标准：

- CPU-only 环境全部成功
- 不允许因 `cupy` 缺失导致非 GPU 功能 import 失败

## 4.2 单元层回归

建议最少覆盖以下对象：

- expression evaluator
- piecewise evaluator
- source term resolver
- reference solution loader
- operator compiler
- operator binder
- derivative analyzer
- method selector
- fallback policy

通过标准：

- 关键输入类型全覆盖
- 异常路径有明确断言
- shape、dtype、返回协议有断言

## 4.3 组件层回归

必须覆盖：

- data generator
- operator factory
- auto code generation
- linear solver backend
- visualization core

通过标准：

- 组件可独立初始化
- 对典型输入输出稳定
- 错误路径不会静默失败

## 4.4 集成层回归

建议覆盖最小完整流程：

- func fitting 最小流程
- linear PDE 最小流程
- nonlinear PDE 最小流程

通过标准：

- 从 case 配置到 solver 初始化可跑通
- 不要求长时间训练，但必须完成 smoke path

## 4.5 案例层回归

建议固定 3 组样例：

- 函数拟合 case
- 线性 PDE case
- 非线性 PDE case

通过标准：

- 结果目录正常创建
- 关键输出文件存在
- 运行过程没有致命异常

## 5. 分阶段回归清单

## 5.1 PR-0 回归基线建设

必须执行：

- 导入层回归
- pytest 收集测试
- 当前已有测试的可运行性扫描

必须新增：

- import smoke tests
- CPU-only smoke tests

## 5.2 PR-1 导入链解耦与 GPU optional 化

必须执行：

- `import src`
- `import src.visualization`
- `pytest --collect-only`
- 可视化相关测试收集
- `LinearSolver` CPU 路径 smoke test

必须验证：

- 无 `cupy` 环境可运行
- 非 GPU 模块不受 GPU 依赖影响

## 5.3 PR-2 抽象层净化与示例/测试剥离

必须执行：

- 原 `operator_usage_example` 关键场景替代测试
- operator 结果一致性测试
- pytest 自动发现新迁移的测试

必须验证：

- `src/abstract_class/` 中不再保留测试型函数
- 示例能力未丢失

## 5.4 PR-3 数据层职责拆解

必须执行：

- 表达式求值测试
- piecewise 测试
- source term 测试
- 自定义 case 模块测试
- 3 类 solver data generator smoke test

必须验证：

- legacy 配置不被破坏
- shape 与默认值行为不变

## 5.5 PR-4 算子构造器重构

必须执行：

- 线性算子单元测试
- 非线性算子单元测试
- 常量与数学函数测试
- 非法表达式错误测试
- operator benchmark

必须验证：

- shape 协议不变
- 关键数值结果在容差内一致
- 错误不再被静默吞掉

## 5.6 PR-5 元编程层重构

必须执行：

- equation parse 测试
- derivative analysis 测试
- code rendering 测试
- golden file 测试
- autocode integration 测试

必须验证：

- 生成产物结构稳定
- 典型 case 生成结果可用

## 5.7 PR-6 线性求解器后端化

必须执行：

- direct/svd/qr/ridge/lsqr/lsmr 正确性测试
- fallback 测试
- CPU-only 回归
- GPU-optional 回归
- 性能对比

必须验证：

- 各方法解在容差内一致
- fallback 路径可追踪
- CPU/GPU 路径都可解释

## 5.8 PR-7 入口层与注册体系整理

必须执行：

- CLI smoke tests
- registry 测试
- autocode 流程测试
- 结果目录测试

必须验证：

- CLI 行为可预期
- 错误 problem type 有清晰异常
- 入口层不再依赖隐式副作用

## 6. 建议测试文件映射

建议新增或整理如下测试：

- `test/test_import_smoke.py`
- `test/test_cpu_only_smoke.py`
- `test/test_operator_usage_examples.py`
- `test/test_base_data_helpers.py`
- `test/test_operator_factory.py`
- `test/test_expression_evaluator.py`
- `test/test_piecewise_evaluator.py`
- `test/test_source_term_resolver.py`
- `test/test_equation_processor.py`
- `test/test_linear_solver_methods.py`
- `test/test_solver_registry.py`
- `test/test_autocode_integration.py`

## 7. 代表性 smoke case 建议

建议固定以下案例作为回归样本：

- [`cases/func_fitting_cases/case_2d`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/func_fitting_cases/case_2d)
- [`cases/linear_pde_cases/poisson_2d_sinpixsinpiy`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/linear_pde_cases/poisson_2d_sinpixsinpiy)
- [`cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D)

如后续发现这些 case 过重，可再补充一组更轻量的最小 case。

## 8. benchmark 建议

以下场景建议建立固定 benchmark：

- operator factory 典型线性算子
- operator factory 典型非线性算子
- linear solver 中小型稠密矩阵
- linear solver 中大型稀疏矩阵
- autocode 生成一次完整 case 的耗时

benchmark 输出至少包含：

- 输入规模
- 执行时间
- 是否使用 GPU
- 与重构前版本的对比

## 9. 合并前检查表

每次提交 PR 前，负责人必须确认：

- 已运行与本 PR 对应的回归项
- 新增测试已纳入仓库
- 文档已更新
- 已记录已知风险和未覆盖点
- 若存在未通过的非关键测试，必须明确说明原因

## 10. 最终验收标准

整个架构优化完成后，应满足：

- CPU-only 环境可稳定导入、测试、运行基础流程
- GPU 路径具备明确 optional 能力
- 顶级热点文件复杂度显著下降
- 关键 case smoke test 稳定通过
- 代码生成具备 golden file 级校验
- 入口层、抽象层、后端层职责边界清晰
