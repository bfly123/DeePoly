# AC Equation Torch From Scratch

这个算例用于验证 Allen-Cahn 伪时间流程在纯 Torch Spotter 下从零开始运行：

- 不使用外部 `spotter_feature_backend`
- `initial_condition.method = given`
- Spotter 由当前框架内部 Torch 网络从随机权重开始训练
- Snipper 使用真实 pseudo-time 质量项推进

建议运行命令：

```bash
conda activate ML
python -m src.cli cases/nonlinear_pde_cases/pseudotime_pde_cases/AC_equation_torch_from_scratch --skip-autocode
```
