# 1D AC Time-PDE Point Scaling

## Scope

Comparison cases:

- `cases/nonlinear_pde_cases/time_pde_cases/AC_equation_imex_rk222_1d_50pts_3seg`
- `cases/nonlinear_pde_cases/time_pde_cases/AC_equation_imex_rk222_1d_100pts_3seg`
- `cases/nonlinear_pde_cases/time_pde_cases/AC_equation_imex_rk222_1d_200pts_3seg`
- `cases/nonlinear_pde_cases/time_pde_cases/AC_equation_imex_rk222_1d_400pts_3seg`

Unified settings used for comparison:

- `n_segments = [3]`
- `dt = 0.04`
- `T = 1.0`
- `time_scheme = IMEX_RK_2_2_2`
- boundary points: left/right each `1`
- `points_per_swap = 1`
- `points_domain_test = [1024]`
- `spotter_skip = 2`
- `epochs_adam = 2000`
- `DNNtol = 1e-4`

Boundary clarification:

- In the current 1D data generator, named boundaries `left` and `right` always map to exactly one coordinate each.
- Therefore, increasing `boundary_conditions[*].points` does not create more distinct 1D boundary locations.
- Boundary enforcement strength should be adjusted through boundary weights, not by trying to add more 1D boundary points.
- The dedicated high-accuracy regression case confirms that `1` point on each side is sufficient to recover the historical best global spacetime error when weights and time-step settings are appropriate.

Special note:

- The `50`-point case uses a smaller Spotter network `hidden_dims = [12, 24, 24, 8]` to keep the linear system overdetermined.
- The `100/200/400` cases use `hidden_dims = [12, 32, 32, 16]`.

## Results

| Points | Physical time (s) | Global spacetime L2 | Global spacetime L∞ |
| --- | ---: | ---: | ---: |
| 50  | 16.95 | 4.040495e-02 | 8.131674e-01 |
| 100 | 21.46 | 3.535081e-02 | 1.314855e-01 |
| 200 | 24.35 | 3.535901e-02 | 1.318892e-01 |
| 400 | 23.27 | 3.535885e-02 | 1.308789e-01 |

## Takeaways

- The main accuracy jump happens from `50 -> 100` points.
- From `100 -> 200 -> 400`, the global spacetime L2 error is almost unchanged under the current time-step and model settings.
- This indicates the current configuration is no longer dominated by spatial sampling density once `100` points are reached.
- If further accuracy is required, reducing `dt` or tightening the Spotter approximation is more promising than continuing to increase points from `100` to `400`.
