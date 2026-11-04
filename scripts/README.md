# Scripts

Every result presented in the report, poster or slides is produced by a script
in this folder. Each script reads its parameters from the matching TOML file in
`configs/`, writes raw data to `results/data/` (via `rydberg.io`) and figures to
`results/figures/` (via `rydberg.plotting`).

Run a single script with

```bash
uv run python scripts/<name>.py --config configs/<name>.toml
```

or all of them with `make figures`.

| Script | Config | Output | Used in |
|---|---|---|---|
| `blockade_dynamics.py` | `configs/blockade_dynamics.toml` | `results/figures/blockade_dynamics.{pdf,png}` | Report Fig. TBD, poster |
| `analytic_residuals.py` | `configs/analytic_residuals.toml` | `results/figures/analytic_residuals.{pdf,png}` | Report Fig. TBD (validation) |
| `detuning_sweep.py` | `configs/detuning_sweep.toml` | `results/figures/detuning_sweep.{pdf,png}` | Report Fig. TBD, poster |
| `sweep_profiles.py` | `configs/sweep_profiles.toml` | `results/figures/sweep_profiles.{pdf,png}` | Report Fig. TBD |
| `timestep_convergence.py` | `configs/timestep_convergence.toml` | `results/figures/timestep_convergence.{pdf,png}` | Report Fig. TBD (validation) |

Extension scripts live in `scripts/extension/` with configs in `configs/extension/`.
