# Time PDE Cases

This directory contains two groups of time-dependent benchmark cases:

- Current tuned local regression cases, including both `IMEX_RK_2_2_2` and `onestep_predictor`
- Imported legacy GitHub benchmark cases, mainly based on `onestep_predictor`

Important note:

- The current `time_pde` framework default `time_scheme` is `onestep_predictor`
- The pseudo-time solver keeps `time_scheme = implicit_euler` as its fixed backend
- Existing tuned local cases may still explicitly set `IMEX_RK_2_2_2` when they are intended to use that scheme
- The imported legacy GitHub cases keep `time_scheme = onestep_predictor` to preserve their original setup
- These imported cases were adapted to the current directory layout and simplified config style
- Reference data was imported without old `results/` artifacts to keep the case tree runnable and lighter
