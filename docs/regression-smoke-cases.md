# DeePoly 回归 Smoke Cases

## 1. 目的

本文档固定架构重构过程中的代表性案例，用于组件级和集成级 smoke test。

## 2. 推荐样例

### 函数拟合

- [`cases/func_fitting_cases/case_2d`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/func_fitting_cases/case_2d)

### 线性 PDE

- [`cases/linear_pde_cases/poisson_2d_sinpixsinpiy`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/linear_pde_cases/poisson_2d_sinpixsinpiy)

### 非线性 PDE

- [`cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D)

## 3. 使用规则

- 每次涉及入口层、solver registry、数据层、代码生成层改动时，至少选取其中一个案例做 smoke test
- 每次涉及后端、求解流程、迭代逻辑改动时，三类案例都应至少跑一次最小验证
- 若这些案例运行过重，应再补充一组轻量专用回归案例，但不能替代当前三组代表性案例
