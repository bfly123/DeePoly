# DeePoly 回归命令清单

## 1. 导入基线

```bash
python -c "import src"
python -c "import src.visualization"
python -c "import src.problem_solvers"
python -c "import src.meta_coding"
python -c "from src.algebraic_solver import LinearSolver"
python -c "from src.algebraic_solver import FastNewtonSolver"
```

## 2. PR-0 / PR-1 基线测试

```bash
pytest -q test/test_import_smoke.py test/test_cpu_only_smoke.py
pytest --collect-only test/test_dimensional_visualizer.py test/test_substep_1_2.py
```

## 3. PR-2 示例迁移兼容测试

```bash
pytest -q test/test_operator_usage_examples.py
```

## 4. 可视化回归

```bash
pytest -q test/test_dimensional_visualizer.py test/test_substep_1_2.py
```

## 5. 组合回归

```bash
pytest -q test/test_import_smoke.py test/test_cpu_only_smoke.py test/test_dimensional_visualizer.py test/test_substep_1_2.py test/test_operator_usage_examples.py test/test_base_data_helpers.py
```

## 6. 集成回归

```bash
pytest -q test/test_integration_with_solver.py
pytest -q test/test_new_test_2d_visualization.py
pytest -q test/test_pseudotime_visualization.py
```

## 7. 说明

- 无 `cupy` 环境时，以上导入基线和 CPU-only 测试必须通过
- 若涉及 GPU 后端修改，应在具备 `cupy` 的环境下补充 GPU 路径验证
