# New_test_2D Multi-Backend Parallel

这个算例演示新的 `spotter` 多后端并行编排：

- `torch_local`
- `jaxpi_worker`
- `paddlesci_worker`

运行模式：

- `spotter.mode = "parallel_select"`
- selector 使用 `composite` 评分
- 当前综合指标由两部分组成：
  - `relative_l2`
  - `boundary_residual`

运行要求：

- 主 DeePoly 环境需要可运行 `torch`
- `jaxpi_worker` 对应的 Python 环境需要可导入 `jax` 与相关 factory 依赖
- `paddlesci_worker` 对应的 Python 环境需要可导入 `paddle` 与 `ppsci`

推荐命令：

```bash
python -m src.cli cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_multibackend_parallel
```

运行完成后，结果目录会额外生成：

- `spotter_selection_report.json`

其中记录每个候选 backend 的评分与最终选中结果。
