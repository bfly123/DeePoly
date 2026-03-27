# 1D AC Time-PDE Segment Scaling

## Scope

Comparison cases:

- `cases/nonlinear_pde_cases/time_pde_cases/AC_equation_imex_rk222_1d_100pts_3seg`
- `cases/nonlinear_pde_cases/time_pde_cases/AC_equation_imex_rk222_1d_100pts_6seg`
- `cases/nonlinear_pde_cases/time_pde_cases/AC_equation_imex_rk222_1d_100pts_12seg`

Unified settings:

- `points_domain = [100]`
- `points_domain_test = [1024]`
- `dt = 0.04`
- `T = 1.0`
- left/right boundary points each `1`
- `points_per_swap = 1`
- `spotter_skip = 2`
- `epochs_adam = 2000`
- `DNNtol = 1e-4`

Special note:

- To keep the linear systems overdetermined as segments increase, the Spotter width was reduced for higher-segment cases.
- `3 seg`: `hidden_dims = [12, 32, 32, 16]`
- `6 seg`: `hidden_dims = [12, 24, 24, 8]`
- `12 seg`: `hidden_dims = [12, 16, 16, 2]`

## Results

| Segments | Physical time (s) | Global spacetime L2 | Global spacetime L∞ |
| --- | ---: | ---: | ---: |
| 3  | 21.46 | 3.535081e-02 | 1.314855e-01 |
| 6  | 17.95 | 3.536445e-02 | 1.314841e-01 |
| 12 | 49.02 | 3.532747e-02 | 1.314404e-01 |

## Takeaways

- Under the current time-step and training setup, increasing segments from `3 -> 6 -> 12` does not materially improve accuracy.
- The three cases have nearly identical global spacetime errors.
- `12` segments is significantly more expensive than `3` or `6` under the current implementation.
- For this 1D AC case, the current bottleneck is not segment count once `3` segments are already used.

## 400-Point Check

Additional direct comparison at `400` domain points:

| Case | Hidden dims | Physical time (s) | Global spacetime L2 | Global spacetime L∞ |
| --- | --- | ---: | ---: | ---: |
| `400pts_3seg` | `[12, 32, 32, 16]` | 23.27 | 3.535885e-02 | 1.308789e-01 |
| `400pts_6seg` | `[12, 24, 24, 8]` | 19.12 | 3.535864e-02 | 1.314606e-01 |

Additional takeaway:

- `400` points can use `6` segments without becoming underdetermined.
- With a narrower Spotter network, `400pts_6seg` remains stable and keeps essentially the same accuracy as `400pts_3seg`.
- For this fast-mode configuration, `6` segments is therefore feasible, but it still does not provide a meaningful accuracy advantage over `3` segments.
