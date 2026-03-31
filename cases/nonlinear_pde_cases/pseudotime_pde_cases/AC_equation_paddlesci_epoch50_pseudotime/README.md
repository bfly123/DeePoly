运行命令：

```bash
cd /home/bfly/workspace/computeforcfd/混合网络/DeePoly_git
conda activate ML
python -m src.cli cases/nonlinear_pde_cases/pseudotime_pde_cases/AC_equation_paddlesci_epoch50_pseudotime
```

评估命令：

```bash
python tools/evaluate_case_solution.py \
  cases/nonlinear_pde_cases/pseudotime_pde_cases/AC_equation_paddlesci_epoch50_pseudotime
```

说明：

- 本算例使用 `epoch_50.pdparams` 作为外部 Paddle spotter。
- `initial_condition.method = external_spotter`，即伪时间步从外部 spotter 预测场直接起步。
- `problem_type = general_pde`，用于测试伪时间步推进是否比固定点 Picard 更适合当前 AC 问题。
