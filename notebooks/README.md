# Notebooks

Notebooks in this folder are for **exploration only**: quick checks, trying out
parameters and sketching plots.

**Nothing presented in the report, poster or slides comes from a notebook.**
Once an idea works, the reusable code moves into `src/rydberg/`, the parameters
into a TOML file in `configs/`, and the figure is produced by a script in
`scripts/`.

Conventions:

- Copy `00_exploration_template.ipynb` and number new notebooks `NN_short_name.ipynb`.
- Outputs are stripped on commit by `nbstripout`, so notebooks stay small and
  diff cleanly.
- Import from the installed `rydberg` package rather than copying code in.
