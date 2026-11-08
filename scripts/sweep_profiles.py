"""Compare linear and cubic detuning sweeps for preparing the Bell state |Psi+>.

Both sweeps run from -Delta_max to +Delta_max over t_max and cross resonance at
t_max / 2. The cubic sweep of handbook Fig. 5.2 is flat at resonance; the linear
sweep crosses at a constant rate. The figure shows

(a) the two profiles Delta(t);
(b) P(|Psi+>)(t) for both, in the blockade regime and at a weaker V;
(c) the instantaneous (adiabatic) energies of H(t) near resonance, in the
    exchange-symmetric subspace {|00>, |Psi+>, |rr>};
(d) the adiabaticity parameter |<e|dH/dt|g>| / (E_e - E_g)^2 of the followed
    state, which is small when the sweep is slow compared with the gap;
(e) the final Bell-state fidelity |<Psi+|Psi(t_max)>|^2 versus t_max, with the
    Landau-Zener estimate for the linear sweep.

Usage
-----
    uv run python scripts/sweep_profiles.py --config configs/sweep_profiles.toml
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from rydberg.config import load_config
from rydberg.evolution.trotter import evolve_trotter
from rydberg.io import save_results
from rydberg.plotting import save_figure, use_style
from rydberg.protocols import (
    SWEEP_PROFILES,
    TimeDependentHamiltonian,
    adiabaticity_parameter,
    detuning_sweep_hamiltonian,
    instantaneous_spectrum,
    landau_zener_bell_population,
)
from rydberg.states import KET_00, KET_RR, PSI_PLUS, fidelity
from rydberg.units import angular_to_mhz, mhz_to_angular

DEFAULT_CONFIG = Path("configs/sweep_profiles.toml")

BRIGHT_SUBSPACE = np.stack([KET_00, PSI_PLUS, KET_RR], axis=-1)
"""Exchange-symmetric states coupled to |00>, as columns."""

RealArray = NDArray[np.float64]


def sweep_hamiltonian(
    profile: str, omega: float, delta_max: float, t_max: float, phi: float, v: float
) -> TimeDependentHamiltonian:
    """Return H(t) for a named sweep profile (all frequencies in rad/us, times in us)."""
    shape = SWEEP_PROFILES[profile]
    return detuning_sweep_hamiltonian(
        omega, lambda t: shape(t, delta_max, t_max), phi, v
    )


def bell_fidelity_trajectory(
    h: TimeDependentHamiltonian, t_max: float, dt: float, sample: str
) -> tuple[RealArray, RealArray]:
    """Return times (us) and |<Psi+|Psi(t)>|^2 along a sweep starting in |00>.

    ``dt`` is adjusted to t_max / round(t_max / dt) so that it divides t_max.
    """
    n_steps = max(1, round(t_max / dt))
    times, states = evolve_trotter(
        h, KET_00, t_max, t_max / n_steps, sample=sample, vectorised=True  # type: ignore[arg-type]
    )
    return times, np.asarray(fidelity(states, PSI_PLUS), dtype=np.float64)


def compute_fidelity_scan(
    profiles: Sequence[str],
    omega: float,
    delta_max: float,
    phi: float,
    v: float,
    t_max_values: RealArray,
    numerics: dict[str, Any],
) -> tuple[dict[str, RealArray], float]:
    """Return the final Bell fidelity versus t_max for each profile.

    Parameters
    ----------
    profiles : sequence of str
        Names in :data:`rydberg.protocols.SWEEP_PROFILES`.
    omega, delta_max, v : float
        Rabi frequency, sweep amplitude and interaction, in rad/us.
    phi : float
        Laser phase, in rad.
    t_max_values : numpy.ndarray
        Sweep durations, in us.
    numerics : dict
        The ``[numerics]`` table.

    Returns
    -------
    fidelities : dict of str to numpy.ndarray
        Final |<Psi+|Psi(t_max)>|^2 for each profile (dimensionless).
    max_step_change : float
        Largest change in any final fidelity when dt is multiplied by
        ``check_factor`` (a step-doubling error estimate).
    """
    dt = float(numerics["dt_us"])
    factor = int(numerics["check_factor"])
    fidelities: dict[str, RealArray] = {}
    max_change = 0.0
    for profile in profiles:
        values = np.empty(t_max_values.size)
        for i, t_max in enumerate(t_max_values):
            h = sweep_hamiltonian(profile, omega, delta_max, float(t_max), phi, v)
            values[i] = bell_fidelity_trajectory(h, float(t_max), dt, numerics["sample"])[1][-1]
            check = bell_fidelity_trajectory(
                h, float(t_max), factor * dt, numerics["sample"]
            )[1][-1]
            max_change = max(max_change, abs(values[i] - check))
        fidelities[profile] = values
    return fidelities, max_change


def plot_profiles(data: dict[str, Any], cfg: dict[str, Any]) -> Figure:
    """Draw the five-panel comparison figure.

    Parameters
    ----------
    data : dict
        Arrays computed in :func:`main` (times in us, frequencies in MHz).
    cfg : dict
        The full config.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    plot_cfg = cfg["plot"]
    colours = plot_cfg["colours"]
    profiles = cfg["sweep"]["profiles"]
    fig = plt.figure(figsize=tuple(plot_cfg["figsize"]))
    grid = fig.add_gridspec(3, 2)
    ax_d = fig.add_subplot(grid[0, 0])
    ax_p = fig.add_subplot(grid[0, 1])
    ax_e = fig.add_subplot(grid[1, 0])
    ax_a = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[2, :])

    v_label = f"{cfg['interaction']['v_mhz']:g}"
    weak_label = f"{cfg['interaction']['v_weak_mhz']:g}"
    for profile in profiles:
        c = colours[profile]
        ax_d.plot(data["times"], data[f"delta_mhz_{profile}"], color=c, label=profile)
        ax_p.plot(data["times"], data[f"p_psi_{profile}"], color=c,
                  label=rf"{profile}, $V/2\pi = {v_label}$ MHz")
        ax_p.plot(data["times"], data[f"p_psi_weak_{profile}"], color=c,
                  linestyle=plot_cfg["linestyle_weak"], linewidth=0.9,
                  label=rf"{profile}, $V/2\pi = {weak_label}$ MHz")
        energies = data[f"energies_mhz_{profile}"]
        for j in range(energies.shape[1]):
            ax_e.plot(data["spectrum_times"], energies[:, j], color=c,
                      label=profile if j == 0 else None)
        ax_a.semilogy(data["spectrum_times"], data[f"adiabaticity_{profile}"], color=c,
                      label=profile)
        ax_f.semilogx(data["t_max_values"], data[f"fidelity_{profile}"], color=c,
                      label=profile)
    ax_f.semilogx(data["t_max_values"], data["fidelity_lz"], color=plot_cfg["colour_lz"],
                  linestyle=":", linewidth=1.0, label="Landau-Zener (linear)")

    ax_d.set_ylabel(r"$\Delta/2\pi$ (MHz)")
    ax_d.legend(loc="upper left")
    ax_p.set_ylabel(r"$P_{|\Psi^+\rangle}$")
    ax_p.set_ylim(0.0, 1.0)
    ax_p.legend(loc="upper left", fontsize=6.5)
    ax_e.set_ylabel(r"$E/2\pi$ (MHz)")
    ax_e.set_ylim(*cfg["spectrum"]["ylim_mhz"])
    ax_e.legend(loc="lower right")
    ax_a.set_ylabel(r"$|\langle e|\dot H|g\rangle| / (E_e - E_g)^2$")
    ax_a.legend(loc="upper left")
    for ax in (ax_d, ax_p, ax_e, ax_a):
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_xlim(float(data["times"][0]), float(data["times"][-1]))
    ax_f.set_xlabel(r"Sweep duration $t_{\max}$ ($\mu$s)")
    ax_f.set_ylabel(r"$|\langle\Psi^+|\Psi(t_{\max})\rangle|^2$")
    ax_f.set_xlim(float(data["t_max_values"][0]), float(data["t_max_values"][-1]))
    ax_f.set_ylim(0.0, 1.02)
    ax_f.legend(loc="upper left")
    ax_a.set_ylim(*plot_cfg["adiabaticity_ylim"])
    for ax, letter in zip((ax_d, ax_p, ax_e, ax_a, ax_f), "abcde", strict=True):
        ax.set_title(f"({letter})", loc="left")
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    """Run the computation, save the data and write the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    omega = mhz_to_angular(float(cfg["drive"]["omega_mhz"]))
    phi = float(cfg["drive"]["phi"])
    delta_max = mhz_to_angular(float(cfg["sweep"]["delta_max_mhz"]))
    t_max = float(cfg["sweep"]["t_max_us"])
    v = mhz_to_angular(float(cfg["interaction"]["v_mhz"]))
    v_weak = mhz_to_angular(float(cfg["interaction"]["v_weak_mhz"]))
    numerics = cfg["numerics"]
    dt = float(numerics["dt_us"])
    profiles = cfg["sweep"]["profiles"]

    data: dict[str, Any] = {}
    spectrum_times = np.linspace(0.0, t_max, int(cfg["spectrum"]["n_times"]))
    data["spectrum_times"] = spectrum_times
    summary: dict[str, Any] = {"final_p_psi_plus": {}, "min_gap_mhz": {}}
    for profile in profiles:
        h = sweep_hamiltonian(profile, omega, delta_max, t_max, phi, v)
        times, p_psi = bell_fidelity_trajectory(h, t_max, dt, numerics["sample"])
        h_weak = sweep_hamiltonian(profile, omega, delta_max, t_max, phi, v_weak)
        _, p_psi_weak = bell_fidelity_trajectory(h_weak, t_max, dt, numerics["sample"])
        data["times"] = times
        data[f"delta_mhz_{profile}"] = angular_to_mhz(
            SWEEP_PROFILES[profile](times, delta_max, t_max)
        )
        data[f"p_psi_{profile}"] = p_psi
        data[f"p_psi_weak_{profile}"] = p_psi_weak
        energies, _ = instantaneous_spectrum(h, spectrum_times, BRIGHT_SUBSPACE)
        data[f"energies_mhz_{profile}"] = angular_to_mhz(energies)
        data[f"adiabaticity_{profile}"] = adiabaticity_parameter(
            h, spectrum_times, 0, BRIGHT_SUBSPACE, float(cfg["spectrum"]["fd_step_us"])
        )
        summary["final_p_psi_plus"][profile] = {
            "v_mhz": float(p_psi[-1]), "v_weak_mhz": float(p_psi_weak[-1])
        }
        summary["min_gap_mhz"][profile] = float(
            angular_to_mhz(np.min(energies[:, 1] - energies[:, 0]))
        )

    scan = cfg["fidelity_scan"]
    t_max_values = np.logspace(
        np.log10(scan["t_max_min_us"]), np.log10(scan["t_max_max_us"]), scan["n_t_max"]
    )
    t_max_values = np.unique(np.round(t_max_values / dt) * dt)  # multiples of dt
    fidelities, max_change = compute_fidelity_scan(
        profiles, omega, delta_max, phi, v, t_max_values, numerics
    )
    data["t_max_values"] = t_max_values
    data["fidelity_lz"] = landau_zener_bell_population(omega, delta_max, t_max_values)
    for profile in profiles:
        data[f"fidelity_{profile}"] = fidelities[profile]

    # Fidelities at selected durations (computed exactly at those t_max).
    report = []
    for t_rep in scan["report_t_max_us"]:
        row: dict[str, float] = {"t_max_us": float(t_rep)}
        for profile in profiles:
            h = sweep_hamiltonian(profile, omega, delta_max, float(t_rep), phi, v)
            row[profile] = float(
                bell_fidelity_trajectory(h, float(t_rep), dt, numerics["sample"])[1][-1]
            )
        row["landau_zener"] = float(landau_zener_bell_population(omega, delta_max, t_rep))
        report.append(row)
    summary["fidelity_table"] = report
    summary["max_step_doubling_change"] = max_change
    # Shortest scanned t_max beyond which the fidelity never drops below the
    # threshold again (the fidelity is not monotonic in t_max).
    threshold = float(scan["fidelity_threshold"])
    summary["fidelity_threshold"] = threshold
    summary["stays_above_threshold_from_us"] = {}
    for profile in profiles:
        failing = np.flatnonzero(fidelities[profile] < threshold)
        if failing.size == 0:
            onset: float | None = float(t_max_values[0])
        elif failing[-1] + 1 < t_max_values.size:
            onset = float(t_max_values[failing[-1] + 1])
        else:
            onset = None
        summary["stays_above_threshold_from_us"][profile] = onset

    save_results(
        cfg["output"]["data"],
        {k: np.asarray(val) for k, val in data.items()},
        metadata={"config_path": str(args.config), "config": cfg, "summary": summary},
    )

    print(f"V/2pi = {cfg['interaction']['v_mhz']:g} MHz, t_max = {t_max:g} us, "
          f"dt = {dt:g} us ({numerics['sample']})")
    for profile in profiles:
        fp = summary["final_p_psi_plus"][profile]
        print(f"  {profile:6s}: P_Psi+(t_max) = {fp['v_mhz']:.5f} "
              f"(V/2pi = {cfg['interaction']['v_weak_mhz']:g} MHz: {fp['v_weak_mhz']:.5f}); "
              f"min gap / 2pi = {summary['min_gap_mhz'][profile]:.4f} MHz")
    print("Final Bell fidelity vs t_max:")
    print("  t_max (us)  " + "  ".join(f"{p:>8s}" for p in profiles) + "  Landau-Zener")
    for row in report:
        print(f"  {row['t_max_us']:10.2f}  " + "  ".join(f"{row[p]:8.5f}" for p in profiles)
              + f"  {row['landau_zener']:8.5f}")
    print(f"  scanned t_max beyond which fidelity stays >= {threshold:g} "
          f"(None: not within the scan): {summary['stays_above_threshold_from_us']}")
    for profile in profiles:
        f = fidelities[profile]
        print(f"  {profile:6s}: max fidelity in scan {np.max(f):.5f} at "
              f"t_max = {t_max_values[np.argmax(f)]:.3f} us; minimum for t_max >= 1 us "
              f"{np.min(f[t_max_values >= 1.0]):.4f}")
    print(f"  step-doubling check: max change in final fidelity = {max_change:.2e}")

    use_style()
    fig = plot_profiles(data, cfg)
    pdf_path, _ = save_figure(fig, cfg["output"]["figure"])
    print(f"Wrote {pdf_path} and {cfg['output']['data']}.npz")


if __name__ == "__main__":
    main()
