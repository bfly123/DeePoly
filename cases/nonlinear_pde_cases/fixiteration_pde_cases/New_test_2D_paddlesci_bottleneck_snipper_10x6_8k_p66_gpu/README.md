# New_test_2D Paddle Bottleneck Snipper 10x6 8k p66 GPU

这个算例只增强 `snipper` 的局部多项式基，不改 `spotter`：

- `spotter bottleneck_size = 20`
- `poly_degree = [6, 6]`
  - 局部多项式基为 `7 x 7 = 49`
- `n_segments = [10, 6]`
- `points_domain = 8000`
- `linear_method = "gpu_direct"`
- `linear_device = "cuda"`

建议命令：

```bash
conda activate ML
python -m src.cli cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_bottleneck_snipper_10x6_8k_p66_gpu
```
