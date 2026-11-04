# Quantum dynamics and quantum computing with Rydberg atoms

[![CI](https://github.com/RonEllenbogen/Quantum-Computing_Computing-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/RonEllenbogen/Quantum-Computing_Computing-Project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains my numerical simulation of two driven, interacting
Rydberg-atom qubits. Each atom is modelled as a two-level system with a ground
state $|0\rangle$ and a Rydberg state $|r\rangle$, coupled by a laser. When both
atoms are excited, the strong van der Waals interaction shifts the doubly
excited state out of resonance. This is the *Rydberg blockade*, the mechanism
behind entangling gates in neutral-atom quantum computers. I solve the
time-dependent Schrödinger equation for this system, validate the solvers
against analytic limits and convergence tests, and use them to study blockade
dynamics and adiabatic state preparation.

## Physics background

Under the rotating-wave approximation, a laser with Rabi frequency $\Omega$,
detuning $\Delta$ and phase $\phi$ drives each atom. Two atoms separated by $R$
experience a van der Waals shift $\hbar V = C_6/R^6$ when both are in
$|r\rangle$. In the basis $\{|00\rangle, |0r\rangle, |r0\rangle, |rr\rangle\}$:

<!-- TODO: verify signs against handbook Eq. (5.4) -->

$$
\frac{H}{\hbar} = \frac{1}{2}
\begin{pmatrix}
2\Delta & \Omega e^{-i\phi} & \Omega e^{-i\phi} & 0 \\
\Omega e^{i\phi} & 0 & 0 & \Omega e^{-i\phi} \\
\Omega e^{i\phi} & 0 & 0 & \Omega e^{-i\phi} \\
0 & \Omega e^{i\phi} & \Omega e^{i\phi} & -2\Delta + 2V
\end{pmatrix}
$$

For $|V| \gg |\Omega|$ the state $|rr\rangle$ is never populated. Starting from
$|00\rangle$, the system then oscillates between $|00\rangle$ and the entangled
Bell state $|\Psi^+\rangle = (|0r\rangle + |r0\rangle)/\sqrt{2}$, with the Rabi
frequency enhanced by $\sqrt{2}$.

## Numerical methods

- **Time-independent Hamiltonians: eigendecomposition.** Diagonalising
  $H = R^\dagger E R$ gives the exact propagator
  $|\Psi(t)\rangle = R^\dagger D(t) R\, |\Psi(0)\rangle$ with
  $D(t) = \mathrm{diag}(e^{-iE_n t})$. I use it for blockade dynamics as a
  function of the interaction strength $V$.
- **Time-dependent Hamiltonians: Trotter–Suzuki stepping.** The state is
  advanced as $|\Psi(t+\delta t)\rangle = e^{-iH(t)\,\delta t}|\Psi(t)\rangle$.
  I use it for adiabatic detuning sweeps such as
  $\Delta(t) = 8\Delta_\text{max}\left[(t - t_\text{max}/2)/t_\text{max}\right]^3$,
  and I determine the step size $\delta t$ needed for a target accuracy
  through a convergence study.

## Repository layout

```
.
├── src/rydberg/        # library code: states, Hamiltonians, propagators, analysis
│   ├── evolution/      # eigendecomposition and Trotter–Suzuki propagators
│   ├── extension/      # extension work
│   └── styles/         # shared Matplotlib style sheet
├── scripts/            # one script per presented result
├── configs/            # one TOML parameter file per script
├── tests/              # unit tests, analytic limits, convergence checks
├── notebooks/          # exploration only
├── results/
│   ├── data/           # raw outputs (.npz + .json metadata, not tracked)
│   └── figures/        # final figures (PDF for LaTeX, PNG for this README)
└── reports/            # LaTeX sources for the report, poster and slides
```

Library code lives in `src/rydberg/`. Every presented result is produced by a
script in `scripts/` that reads a config in `configs/`. Notebooks are for
exploration only.

## Installation and quickstart

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone https://github.com/RonEllenbogen/Quantum-Computing_Computing-Project.git
cd Quantum-Computing_Computing-Project
make setup        # uv sync + pre-commit and nbstripout hooks
make figures      # run every script in scripts/
```

Run a single result with

```bash
uv run python scripts/blockade_dynamics.py --config configs/blockade_dynamics.toml
```

Run `make help` to list all tasks (`test`, `lint`, `typecheck`, `report`, ...).

## Reproducing results

| Figure | Script | Config |
|---|---|---|
| Rydberg-blockade dynamics vs $V$ | `scripts/blockade_dynamics.py` | `configs/blockade_dynamics.toml` |
| Residuals against analytic limits | `scripts/analytic_residuals.py` | `configs/analytic_residuals.toml` |
| Adiabatic cubic detuning sweep | `scripts/detuning_sweep.py` | `configs/detuning_sweep.toml` |
| Linear vs cubic sweep profiles | `scripts/sweep_profiles.py` | `configs/sweep_profiles.toml` |
| Time-step convergence | `scripts/timestep_convergence.py` | `configs/timestep_convergence.toml` |

Each script saves its raw data to `results/data/` with metadata (config, git
commit, timestamp, package version) and its figures to `results/figures/`.

## Testing and validation

A simulation is only useful if it can be trusted, so validation is treated as
a result in its own right:

- **Analytic limits.** Numerical evolution is compared against closed-form
  solutions for non-interacting atoms ($V = 0$) and for perfect blockade
  ($|V| \gg |\Omega|$), and the residuals are reported.
- **Convergence.** The Trotter–Suzuki error is measured as a function of
  $\delta t$ to find the largest step that gives four significant figures.
- **Conservation laws.** Norm conservation, unitarity of the propagators and
  hermiticity of the Hamiltonians are checked in the test suite.

```bash
make test         # full suite with coverage
make test-fast    # skip tests marked slow
```

## Unit conventions

Throughout the code: $\hbar = 1$, **time in μs**, and **frequencies as angular
frequencies in rad/μs**. A Rabi frequency quoted as $\Omega/2\pi = 1$ MHz is
therefore `Omega = 2 * np.pi * 1.0` in the code.

## Roadmap

The milestone is built bottom-up. Each module gets tests before the next one
builds on it.

**Foundations**

- [x] 1. `units.py`: conversions between MHz and rad/μs.
- [x] 2. `states.py`: the basis states, $|\Psi^+\rangle$, populations, norm
      and fidelity. Tests in `test_states.py`.
- [x] 3. `hamiltonians.py`: single-atom and two-atom Hamiltonians. Tests in
      `test_hamiltonians.py` cover hermiticity, the matrix elements, and the
      $V = 0$ limit reducing to two independent atoms.

**Part 1: time-independent Hamiltonian**

- [ ] 4. `analytic.py`: closed-form solutions for $V = 0$ and for perfect
      blockade ($|V| \gg |\Omega|$, a $\sqrt{2}\,\Omega$ oscillation between
      $|00\rangle$ and $|\Psi^+\rangle$). Tests in `test_analytic.py`.
- [ ] 5. `evolution/eigen.py`: the eigendecomposition propagator. Tests in
      `test_evolution.py` check norm conservation, unitarity and agreement
      with `scipy.linalg.expm`.
- [ ] 6. `convergence.py`: error metrics such as residuals and the norm error.
- [ ] 7. `configs/` + `scripts/blockade_dynamics.py`: reproduce the blockade
      plot for $\Omega/2\pi = 1$ MHz and $V/2\pi$ from 0 to 100 MHz.
- [ ] 8. `scripts/analytic_residuals.py`: residuals against both analytic
      limits.

**Part 2: time-dependent Hamiltonian**

- [ ] 9. `protocols.py`: linear and cubic detuning sweeps $\Delta(t)$.
- [ ] 10. `evolution/trotter.py`: Trotter–Suzuki stepping. Test that it
      reproduces the eigendecomposition result for a constant Hamiltonian.
- [ ] 11. `scripts/timestep_convergence.py`: find the largest $\delta t$ that
      gives four significant figures. Mark its tests `convergence` and, if
      long, `slow`.
- [ ] 12. `scripts/detuning_sweep.py`: the cubic sweep with
      $\Delta_\text{max}/2\pi = 50$ MHz and $t_\text{max} = 1$ μs, run at the
      converged $\delta t$.
- [ ] 13. `scripts/sweep_profiles.py`: compare the linear and cubic profiles.

**Write-up**

- [ ] 14. Embed the key figures under [Results](#results) and fill in the
      report's `validation.tex` and `results.tex`.
- [ ] 15. Choose and scope the extension.

## Results

<!-- Embed the PNGs from results/figures/ here once the scripts are written. -->

## Extension

To be decided.

## References

1. S. Jandura and G. Pupillo, "Time-Optimal Two- and Three-Qubit Gates for
   Rydberg Atoms", *Quantum* **6**, 712 (2022).
   [doi:10.22331/q-2022-05-13-712](https://doi.org/10.22331/q-2022-05-13-712)
2. S. J. Evered *et al.*, "High-fidelity parallel entangling gates on a
   neutral-atom quantum computer", *Nature* **622**, 268 (2023).
   [doi:10.1038/s41586-023-06481-y](https://doi.org/10.1038/s41586-023-06481-y)
3. H. Bernien *et al.*, "Probing many-body dynamics on a 51-atom quantum
   simulator", *Nature* **551**, 579 (2017).
   [doi:10.1038/nature24622](https://doi.org/10.1038/nature24622)

## License

Released under the [MIT License](LICENSE).

## Context

Developed for the PHYS3561 Computing Project, Department of Physics, Durham
University.
