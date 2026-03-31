# New_test_2D Paddle Bottleneck Snipper 10x6 8k p55 GPU

这是 `10x6 / 5k / p55 / gpu_direct` 的加点数对照算例：

- `n_segments = [10, 6]`
- `poly_degree = [5, 5]`
- `points_domain = 8000`
- `snipper_sampling.points_domain = 8000`
- `linear_method = "gpu_direct"`
- `linear_device = "cuda"`

建议命令：

```bash
conda activate ML
python -m src.cli cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_bottleneck_snipper_10x6_8k_p55_gpu
```
