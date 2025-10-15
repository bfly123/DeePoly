# PDE Solver 重构计划

**日期**: 2025-10-10
**目标**: general_pde → pseudotime_pde 重命名 + 新增 fixiteration_pde

---

## 📋 Phase 1: 重命名 general_pde_solver → pseudotime_pde_solver ✅

### Task 1.1: 创建新目录结构
- [x] 创建 `src/problem_solvers/pseudotime_pde_solver/`
- [x] 创建 `src/problem_solvers/pseudotime_pde_solver/core/`
- [x] 创建 `src/problem_solvers/pseudotime_pde_solver/utils/`

### Task 1.2: 复制并重命名核心文件
- [x] solver.py: GeneralPDESolver → PseudoTimePDESolver
- [x] core/fitter.py: GeneralPDEFitter → PseudoTimePDEFitter
- [x] core/net.py: GeneralPDENet → PseudoTimePDENet
- [x] core/spotter_init.py: 保持SpotterNet不变

### Task 1.3: 复制并重命名工具文件
- [x] utils/config.py: GeneralPDEConfig → PseudoTimePDEConfig
- [x] utils/data.py: GeneralPDEDataGenerator → PseudoTimePDEDataGenerator
- [x] utils/visualize.py: GeneralPDEVisualizer → PseudoTimePDEVisualizer

### Task 1.4: 更新所有 __init__.py
- [x] pseudotime_pde_solver/__init__.py
- [x] pseudotime_pde_solver/core/__init__.py
- [x] pseudotime_pde_solver/utils/__init__.py

### Task 1.5: 更新文档和注释
- [x] 修改所有文件的文档字符串
- [x] 更新注释中的 "general_pde" → "pseudotime_pde"
- [x] 更新注释中的 "General PDE" → "Pseudo-Time PDE"

### Task 1.6: 测试验证
- [x] 测试导入: `from src.problem_solvers.pseudotime_pde_solver import PseudoTimePDESolver`
- [x] 验证功能不变

### Task 1.7: 清理
- [x] 删除旧目录 `src/problem_solvers/general_pde_solver/`

---

## 📋 Phase 2: 创建 fixiteration_pde_solver

### Task 2.1: 创建目录结构
- [x] 创建 `src/problem_solvers/fixiteration_pde_solver/`
- [x] 创建 `src/problem_solvers/fixiteration_pde_solver/core/`
- [x] 创建 `src/problem_solvers/fixiteration_pde_solver/utils/`

### Task 2.2: 实现 FixIterationPDEFitter
- [x] 创建 core/fitter.py
- [x] 继承 BaseDeepPolyFitter
- [x] 实现 fit(U_current) 方法
- [x] 实现 solve_steady_state() 方法

### Task 2.3: 实现 FixIterationPDESolver
- [x] 创建 solver.py
- [x] 继承 NonlinearPDESolver
- [x] 实现 solve() 使用统一迭代框架
- [x] 实现所有工厂方法

### Task 2.4: 实现配置和工具类
- [x] utils/config.py: FixIterationPDEConfig
- [x] utils/data.py: 复用或继承 PseudoTimePDEDataGenerator
- [x] utils/visualize.py: 复用或继承 PseudoTimePDEVisualizer
- [x] core/net.py: 复用 PseudoTimePDENet

### Task 2.5: 创建 __init__.py
- [x] fixiteration_pde_solver/__init__.py
- [x] fixiteration_pde_solver/core/__init__.py
- [x] fixiteration_pde_solver/utils/__init__.py

### Task 2.6: 测试验证
- [x] 测试导入
- [x] 创建简单测试案例 (fixpoint_sine_1d)
- [x] 验证不动点迭代功能（8次迭代收敛，误差 < 2e-2）

---

## 📋 Phase 3: 更新文档

### Task 3.1: 更新设计文档
- [ ] 更新 NONLINEAR_SOLVER_UNIFIED_ITERATION.md
- [ ] 更新 NONLINEAR_SOLVER_IMPLEMENTATION_ANALYSIS.md
- [ ] 更新 IMPLEMENTATION_SUMMARY.md

### Task 3.2: 创建迁移指南
- [ ] 创建 MIGRATION_GUIDE.md
- [ ] 说明 general_pde → pseudotime_pde 的变更
- [ ] 提供代码迁移示例

---

## 当前状态

- [x] NonlinearPDESolver 基类完成
- [x] 统一迭代框架实现
- [x] 通用初始化方法实现
- [x] Phase 1: 重命名 general_pde → pseudotime_pde ✅
- [x] Phase 2: 创建 fixiteration_pde_solver ✅
  - [x] FixIterationPDEFitter 实现
  - [x] FixIterationPDESolver 实现
  - [x] 边界约束支持（局部 Dirichlet 约束）
  - [x] 非线性项正确处理（修复列表到数组转换）
  - [x] 特征列表正确传递（支持含导数的非线性项）
  - [x] 测试案例通过（Allen-Cahn 1D，7次迭代，误差 ~1e-5）
- [ ] Phase 3: 更新文档

---

## 执行原则

1. **代码简洁**: 避免复杂的条件分支
2. **根本安全**: 通过设计保证安全，不依赖运行时检查
3. **功能导向**: 只实现需要的功能，不过度发挥
4. **逐步验证**: 每个Task完成后验证功能
