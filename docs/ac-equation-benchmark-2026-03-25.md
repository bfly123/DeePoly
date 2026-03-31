# AC Equation Benchmark Notes

## Scope

Case directory:

- `cases/nonlinear_pde_cases/pseudotime_pde_cases/AC_equation`

Reference data:

- `reference_data/allen_cahn_highres.mat`

Global error metrics were computed by:

1. loading `solution_data.npz`
2. rebuilding the numerical structured grid from `x_test`
3. interpolating numerical `spotter_test` / `sniper_test` onto the reference grid
4. comparing against the full high-resolution reference field `usol`

Reference grid size:

- `x = 2048`
- `t = 1001`

Reported metrics:

- `relL2`
- `RMSE`
- `Linf`

## Code Fixes Applied

### 1. Spotter best-model checkpoint fix

File:

- `src/abstract_class/base_net_training_mixin.py`

Problem:

- the previous implementation only saved the model at the end of each retry
- if Adam reached a good minimum in the middle of training and later bounced away, the final worse model was restored

Fix:

- best model is now tracked inside Adam / LBFGS loops
- the globally best state over the whole run is restored at the end

Impact:

- this removed a real regression in training-state selection
- example: for `3x3 / 9x6 / 10000 / 200x200`, the selected training loss changed from a bad final value around `3.424` back to the true in-run best around `1.652e-03`

### 2. Boundary-condition weight propagation fix

Files:

- `src/abstract_class/config/base_data_boundary_mixin.py`
- `src/abstract_class/config/base_data_boundary_segment_processing_mixin.py`
- `src/abstract_class/boundary_condition_loss_mixin.py`

Problem:

- per-boundary `weight` in `config.json` existed but was not actually used in NN boundary loss

Fix:

- boundary weights are now stored, moved to GPU, and used in weighted boundary loss accumulation

Note:

- this is semantically correct, but for the current AC case the existing config heavily favors left/right boundaries over the bottom initial-condition boundary, so the case now needs retuning if those weights are to remain active

### 3. AC sweep tool enhancement

File:

- `tools/sweep_ac_equation.py`

Enhancement:

- supports structured training grids in `points_domain`, for example `80x80,100x100,120x120`
- summary CSV now includes both the raw grid spec and materialized total point count

## Current Reproducible Results

### A. Single-step random baseline, current code

Run:

- `sweeps/smoke_after_bc_fix/runs/s3x3_p9x6_d10000_e10000_g200x200`

Config:

- `n_segments = 3x3`
- `poly_degree = 9x6`
- `points_domain = 10000` random points
- `epochs_adam = 10000`
- `points_domain_test = 200x200`
- `pseudo_time.max_iterations = 1`

Metrics:

- `spotter`: `relL2 = 0.827451`, `RMSE = 0.586993`, `Linf = 1.945999`
- `sniper`: `relL2 = 0.630904`, `RMSE = 0.447562`, `Linf = 1.526008`

### B. Structured training-grid comparison

Summary file:

- `cases/nonlinear_pde_cases/pseudotime_pde_cases/AC_equation/sweeps/structured_grid_round1/summary.csv`

Tested:

- `3x3 / 9x6 / 80x80`
- `3x3 / 9x6 / 100x100`
- `3x3 / 9x6 / 120x120`

Results:

| Train grid | Total points | Spotter relL2 | Sniper relL2 |
| --- | ---: | ---: | ---: |
| `80x80` | 6400 | `0.828049` | `0.630341` |
| `100x100` | 10000 | `0.829745` | `0.630924` |
| `120x120` | 14400 | `0.834114` | `0.631230` |

Conclusion:

- under the current code path, `80x80` is the best structured training grid among the tested options
- increasing the structured train grid size did not improve global error

### C. Segment-grid comparison

Summary file:

- `cases/nonlinear_pde_cases/pseudotime_pde_cases/AC_equation/sweeps/segment_grid_round1/summary.csv`

Fixed settings:

- `poly_degree = 9x6`
- `points_domain = 80x80`
- `epochs_adam = 10000`
- `points_domain_test = 200x200`
- `pseudo_time.max_iterations = 1`

Results:

| Segment grid | Spotter relL2 | Sniper relL2 | Sniper RMSE | Sniper Linf |
| --- | ---: | ---: | ---: | ---: |
| `2x2` | `0.830687` | `0.630346` | `0.447166` | `1.518373` |
| `3x3` | `0.828049` | `0.630341` | `0.447163` | `1.517972` |
| `4x4` | `0.830616` | `0.630338` | `0.447161` | `1.518270` |

Conclusion:

- best final `sniper` result in this round is `4x4 / 80x80 / 9x6`
- however the difference between `2x2`, `3x3`, and `4x4` is negligible at the fourth decimal place
- from a cost / simplicity standpoint, `3x3 / 80x80 / 9x6` is already effectively at the same level

## Multi-Step Pseudo-Time Regression Test

### 1. `max_iterations = 10`, adaptive `dt`

Run directory:

- `cases/nonlinear_pde_cases/pseudotime_pde_cases/AC_equation/sweeps/manual_runs/iter10_base`

Observed behavior:

- iteration 1: `Res ≈ 6.28e-01`
- iteration 2: `Res ≈ 1.24e+00`
- iteration 3: `Res ≈ 1.01e+00`, `dU_max ≈ 2.39e+03`
- iteration 4: linear solve residual became extremely large, `dU_max ≈ 1.71e+09`

Conclusion:

- multi-step pseudo-time stepping is unstable for this case in the current framework
- allowing `dt` to increase to `0.11` makes the divergence appear even faster

### 2. `max_iterations = 5`, fixed `dt = 0.05`

Run directory:

- `cases/nonlinear_pde_cases/pseudotime_pde_cases/AC_equation/sweeps/manual_runs/iter5_dt005_fixed`

Convergence info:

- `converged = false`
- `reason = max_iterations`
- residual history tail:
  `0.627663 -> 1.235295 -> 1.011825 -> 1.000000 -> 1.000000`

Metrics of the saved final field:

- `spotter`: `relL2 = 0.543301`, `RMSE = 0.385417`, `Linf = 1.531900`
- `sniper`: `relL2 = 0.630904`, `RMSE = 0.447562`, `Linf = 1.526008`

Conclusion:

- reducing to a fixed `dt = 0.05` does not solve the instability
- the current pseudo-time iteration chain still fails to converge for AC beyond the first step

## Legacy Stored Result Comparison

Legacy file:

- `cases/nonlinear_pde_cases/pseudotime_pde_cases/AC_equation/results/solution_data.npz`

Computed metrics:

- `spotter`: `relL2 = 0.054821`, `RMSE = 0.038890`, `Linf = 0.609917`
- `sniper`: `relL2 = 0.053149`, `RMSE = 0.037704`, `Linf = 0.651058`

Interpretation:

- the old stored result is still much better than any current reproducible rerun
- that legacy quality has not yet been recovered in the current framework

## Main Conclusions

1. The training checkpoint selection bug was real and is now fixed.
2. The AC case can be benchmarked reproducibly in single-step mode.
3. Under the current code path, the best tested single-step final result is approximately:
   `sniper relL2 ≈ 0.63034`
4. Structured training grids do not materially improve the current AC result.
5. Segment-grid refinement from `2x2` to `4x4` also does not materially improve the current AC result.
6. The dominant unresolved issue is not single-step training, but instability of the multi-step pseudo-time iteration chain for this case.
7. The legacy `~0.053` global relative L2 level remains an open regression target.

## Recommended Next Actions

1. Compare the current AC pseudo-time update path against the older stable AC implementation step by step, especially:
   linear system assembly,
   coefficient reuse,
   residual definition,
   and `dt` update logic.
2. Add an AC-specific regression test that checks:
   single-step reproducibility,
   and that multi-step residual does not increase catastrophically after step 2.
3. Revisit AC boundary weighting now that per-BC `weight` is actually honored; the current config likely over-constrains side boundaries relative to the bottom initial-condition boundary.
