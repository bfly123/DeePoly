# New_test_2D Paddle Bottleneck Snipper 10x6 8k p55 GPU SVD

这个算例是 `10x6 / 8k / p55 / float64 snipper` 的 GPU-SVD 对照版本：

- `spotter` 不变，仍为 `20` 维 bottleneck
- `poly_degree = [5, 5]`
- `n_segments = [10, 6]`
- `points_domain = 8000`
- `linear_method = "gpu_svd"`
- `linear_device = "cuda"`

建议命令：

```bash
conda activate ML
python -m src.cli cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_bottleneck_snipper_10x6_8k_p55_gpu_svd
```
