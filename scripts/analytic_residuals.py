"""Compute residuals of the numerical evolution against analytic limits.

Two kinds of residual are separated:

* **Numerical error.** At V = 0 the analytic solution is exact, so the residual
  of the eigendecomposition propagator measures only floating-point round-off.
* **Model (systematic) error.** For V != 0 the residual against the V = 0
  solution, or against the blockade solution, measures how good each
  *approximation* is, not the numerics.

The scan over V / Omega shows both limits and the crossover between them. The
fitted power laws are compared with perturbation theory: amplitudes deviate at
first order in V/Omega (or Omega/V), populations at second order at Delta = 0.

Usage
-----
    uv run python scripts/analytic_residuals.py --config configs/analytic_residuals.toml
"""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from rydberg.analytic import blockade_amplitudes, no_interaction_amplitudes
from rydberg.config import load_config
from rydberg.convergence import fit_power_law
from rydberg.evolution.eigen import evolve
from rydberg.hamiltonians import two_atom_hamiltonian
from rydberg.io import save_results
from rydberg.plotting import save_figure, use_style
from rydberg.states import KET_00, ComplexArray
from rydberg.units import mhz_to_angular

DEFAULT_CONFIG = Path("configs/analytic_residuals.toml")

RealArray = NDArray[np.float64]
AnalyticFn = Callable[[RealArray, float, float, float], ComplexArray]


def amplitude_residual(numerical: ComplexArray, analytic: ComplexArray) -> RealArray:
    """Return max_i |psi_i^num - psi_i^ana| at each time (dimensionless)."""
    result: RealArray = np.max(np.abs(numerical - analytic), axis=-1)
    return result


def population_residual(numerical: ComplexArray, analytic: ComplexArray) -> RealArray:
    """Return max_i | |psi_i^num|^2 - |psi_i^ana|^2 | at each time (dimensionless)."""
    result: RealArray = np.max(np.abs(np.abs(numerical) ** 2 - np.abs(analytic) ** 2), axis=-1)
    return result


def scan_residuals(
    omega: float,
    delta: float,
    phi: float,
    v_values: RealArray,
    times: RealArray,
    analytic_fn: AnalyticFn,
) -> tuple[RealArray, RealArray]:
    """Return the max-over-time residuals against one analytic solution for each V.

    Parameters
    ----------
    omega, delta : float
        Rabi frequency and detuning, in rad/us.
    phi : float
        Laser phase, in rad.
    v_values : numpy.ndarray, shape (n_v,)
        Van der Waals shifts V, in rad/us.
    times : numpy.ndarray, shape (n_times,)
        Evolution times, in us.
    analytic_fn : callable
        Analytic solution ``f(times, omega, delta, phi)`` starting from |00>,
        e.g. :func:`rydberg.analytic.blockade_amplitudes`.

    Returns
    -------
    amplitude, population : numpy.ndarray, shape (n_v,)
        Max over time of the amplitude and population residuals (dimensionless).
    """
    analytic = analytic_fn(times, omega, delta, phi)
    amplitude = np.empty(v_values.size)
    population = np.empty(v_values.size)
    for i, v in enumerate(v_values):
        numerical = evolve(two_atom_hamiltonian(omega, delta, phi, float(v)), KET_00, times)
        amplitude[i] = np.max(amplitude_residual(numerical, analytic))
        population[i] = np.max(population_residual(numerical, analytic))
    return amplitude, population


def time_trace(
    omega: float, delta: float, phi: float, v: float, times: RealArray, analytic_fn: AnalyticFn
) -> RealArray:
    """Return the amplitude residual against ``analytic_fn`` at each time (dimensionless)."""
    numerical = evolve(two_atom_hamiltonian(omega, delta, phi, v), KET_00, times)
    return amplitude_residual(numerical, analytic_fn(times, omega, delta, phi))


def fit_window(
    ratios: RealArray, residuals: RealArray, mask: NDArray[np.bool_]
) -> tuple[float, float]:
    """Fit ``residuals = A * ratios**p`` inside ``mask``; return ``(p, A)``."""
    return fit_power_law(ratios[mask], residuals[mask])


def plot_residuals(
    ratios: RealArray,
    scans: dict[str, RealArray],
    fits: dict[str, float],
    times: RealArray,
    traces: dict[str, RealArray],
    labels: dict[str, str],
    plot_cfg: dict[str, Any],
) -> Figure:
    """Plot (a) residual vs V / Omega and (b) residual vs time.

    Parameters
    ----------
    ratios : numpy.ndarray, shape (n_ratio,)
        V / Omega (dimensionless).
    scans : dict
        Max-over-time residuals keyed ``"{weak,strong}_{amplitude,population}"``.
    fits : dict
        Fitted exponents with the same keys.
    times : numpy.ndarray, shape (n_times,)
        Evolution times, in us.
    traces : dict
        Amplitude residual vs time keyed ``"v0"``, ``"v0_detuned"``, ``"blockade"``.
    labels : dict
        Legend labels for the traces (same keys).
    plot_cfg : dict
        The ``[plot]`` table of the config.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    fig, (ax_v, ax_t) = plt.subplots(2, 1, figsize=tuple(plot_cfg["figsize"]))
    colours = {"weak": plot_cfg["colour_weak"], "strong": plot_cfg["colour_strong"]}
    names = {"weak": r"vs $V{=}0$", "strong": "vs blockade"}

    for regime in ("weak", "strong"):
        colour = colours[regime]
        amp = scans[f"{regime}_amplitude"]
        pop = scans[f"{regime}_population"]
        amp_slope = fits[f"{regime}_amplitude"]
        pop_slope = fits[f"{regime}_population"]
        ax_v.loglog(
            ratios,
            amp,
            color=colour,
            linestyle="--",
            label=f"amplitude {names[regime]} (${amp_slope:+.2f}$)",
        )
        ax_v.loglog(
            ratios, pop, color=colour, label=f"population {names[regime]} (${pop_slope:+.2f}$)"
        )
    ax_v.set_xlabel(r"$V/\Omega$")
    ax_v.set_ylabel("max residual")
    ax_v.set_ylim(bottom=float(plot_cfg["scan_ylim_bottom"]))
    ax_v.legend(
        loc="lower center",
        ncol=2,
        fontsize="xx-small",
        title="fitted log-log slope in brackets",
        title_fontsize="xx-small",
    )
    ax_v.set_title("(a)", loc="left")

    trace_colours = {
        "v0": plot_cfg["colour_weak"],
        "v0_detuned": plot_cfg["colour_v0_detuned"],
        "blockade": plot_cfg["colour_strong"],
    }
    for key, trace in traces.items():
        # Residuals vanish exactly at t = 0, which a log axis cannot show.
        ax_t.semilogy(times[1:], trace[1:], color=trace_colours[key], label=labels[key])
    ax_t.set_xlabel(r"Time ($\mu$s)")
    ax_t.set_ylabel("amplitude residual")
    ax_t.set_xlim(float(times[0]), float(times[-1]))
    ax_t.legend(loc="center right", fontsize="x-small")
    ax_t.set_title("(b)", loc="left")
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    """Run the residual study, save the data and write the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    omega = mhz_to_angular(float(cfg["drive"]["omega_mhz"]))
    delta = mhz_to_angular(float(cfg["drive"]["delta_mhz"]))
    phi = float(cfg["drive"]["phi"])
    times = np.linspace(0.0, cfg["time"]["t_max_us"], cfg["time"]["n_times"])
    scan = cfg["scan"]
    ratios = np.logspace(scan["log10_ratio_min"], scan["log10_ratio_max"], scan["n_ratio"])
    v_values = ratios * omega

    scans: dict[str, RealArray] = {}
    for regime, fn in (("weak", no_interaction_amplitudes), ("strong", blockade_amplitudes)):
        amplitude, population = scan_residuals(omega, delta, phi, v_values, times, fn)
        scans[f"{regime}_amplitude"] = amplitude
        scans[f"{regime}_population"] = population

    masks = {
        "weak": ratios <= scan["weak_fit_max"],
        "strong": ratios >= scan["strong_fit_min"],
    }
    fits: dict[str, float] = {}
    prefactors: dict[str, float] = {}
    for key, values in scans.items():
        slope, prefactor = fit_window(ratios, values, masks[key.split("_")[0]])
        fits[key] = slope
        prefactors[key] = prefactor

    tt = cfg["time_traces"]
    delta_d = mhz_to_angular(float(tt["v0_detuned_delta_mhz"]))
    phi_d = float(tt["v0_detuned_phi"])
    v_blockade = mhz_to_angular(float(tt["blockade_v_mhz"]))
    traces = {
        "v0": time_trace(omega, delta, phi, 0.0, times, no_interaction_amplitudes),
        "v0_detuned": time_trace(omega, delta_d, phi_d, 0.0, times, no_interaction_amplitudes),
        "blockade": time_trace(omega, delta, phi, v_blockade, times, blockade_amplitudes),
    }
    labels = {
        "v0": r"$V=0$ (exact)",
        "v0_detuned": rf"$V=0$, $\Delta/2\pi={tt['v0_detuned_delta_mhz']:g}$ MHz, "
        rf"$\phi={phi_d:g}$",
        "blockade": rf"$V/2\pi={tt['blockade_v_mhz']:g}$ MHz vs blockade",
    }

    summary = {
        "fitted_exponents": fits,
        "fitted_prefactors": prefactors,
        "max_round_off_residual_v0": float(max(np.max(traces["v0"]), np.max(traces["v0_detuned"]))),
        "max_blockade_residual_at_blockade_v": float(np.max(traces["blockade"])),
    }
    save_results(
        cfg["output"]["data"],
        {"times_us": times, "v_over_omega": ratios, **scans, **{f"trace_{k}": v for k, v in traces.items()}},
        metadata={"config_path": str(args.config), "config": cfg, "summary": summary},
    )

    use_style()
    fig = plot_residuals(ratios, scans, fits, times, traces, labels, cfg["plot"])
    pdf_path, _ = save_figure(fig, cfg["output"]["figure"])
    for key, slope in fits.items():
        print(f"{key:22s} slope = {slope:+.3f}")
    print(f"max V = 0 residual (round-off): {summary['max_round_off_residual_v0']:.2e}")
    print(f"Wrote {pdf_path} and {cfg['output']['data']}.npz")


if __name__ == "__main__":
    main()
