"""Reproduce the cubic detuning sweep of handbook Fig. 5.2 with Trotter-Suzuki stepping.

Two atoms start in |00> and are driven with constant Rabi frequency Omega while
the detuning is swept through resonance, Delta(t) = 8 Delta_max
[(t - t_max/2) / t_max]^3. For each van der Waals shift V the time-dependent
Hamiltonian of handbook Eq. (5.4) is propagated with Eq. (5.7), and the
populations of |00> and of the Bell state |Psi+> = (|0r> + |r0>) / sqrt(2) are
recorded.

The figure reproduces handbook Fig. 5.2: Delta(t) on top, then P(|00>) and
P(|Psi+>), with V = 0 in blue and V > 0 in red, opacity increasing with V.

At t_max = 1 us the sweep is *not* adiabatic: the adiabaticity parameter
peaks at ~0.3-0.5 (see ``sweep_profiles.py``), and the final Bell-state
population is not monotonic in t_max. The high P(|Psi+>) at 1 us is partly a
coherent pulse-area effect of this particular duration, not adiabatic
following alone.

Two checks are printed and saved with the data: a step-doubling estimate of the
time-step error over all V, and the final P(|Psi+>) with the sign of Delta
reversed. Delta -> -Delta is not a symmetry for V != 0, so the second shows how
sensitive the comparison with Fig. 5.2 is to the sign convention of Eq. (5.4).

Usage
-----
    uv run python scripts/detuning_sweep.py --config configs/detuning_sweep.toml
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
from rydberg.protocols import SWEEP_PROFILES, DetuningProfile, detuning_sweep_hamiltonian
from rydberg.states import KET_00, PSI_PLUS, overlap_mod_squared
from rydberg.units import angular_to_mhz, mhz_to_angular

DEFAULT_CONFIG = Path("configs/detuning_sweep.toml")

RealArray = NDArray[np.float64]


def run_sweep(
    omega: float,
    detuning: DetuningProfile,
    phi: float,
    v: float,
    t_max: float,
    n_steps: int,
    sample: str,
) -> tuple[RealArray, RealArray, RealArray]:
    """Evolve |00> through one sweep and return the populations.

    Parameters
    ----------
    omega : float
        Rabi frequency Omega, in rad/us.
    detuning : callable
        Detuning profile Delta(t), us -> rad/us.
    phi : float
        Laser phase, in rad.
    v : float
        Van der Waals shift V, in rad/us.
    t_max : float
        Sweep duration, in us.
    n_steps : int
        Number of Trotter steps (dt = t_max / n_steps).
    sample : str
        Sampling rule passed to :func:`evolve_trotter`.

    Returns
    -------
    times : numpy.ndarray, shape (n_steps + 1,)
        Times, in us.
    p_00, p_psi_plus : numpy.ndarray, shape (n_steps + 1,)
        Populations of |00> and |Psi+> (dimensionless).
    """
    h = detuning_sweep_hamiltonian(omega, detuning, phi, v)
    times, states = evolve_trotter(
        h, KET_00, t_max, t_max / n_steps, sample=sample, vectorised=True  # type: ignore[arg-type]
    )
    p_00 = np.asarray(overlap_mod_squared(states, KET_00), dtype=np.float64)
    p_psi = np.asarray(overlap_mod_squared(states, PSI_PLUS), dtype=np.float64)
    return times, p_00, p_psi


def compute_populations(
    omega: float,
    detuning: DetuningProfile,
    phi: float,
    v_values: RealArray,
    t_max: float,
    numerics: dict[str, Any],
    n_saved_times: int,
) -> tuple[RealArray, RealArray, RealArray, float]:
    """Return P(|00>) and P(|Psi+>) for each V, and a step-doubling error estimate.

    Parameters
    ----------
    omega : float
        Rabi frequency Omega, in rad/us.
    detuning : callable
        Detuning profile Delta(t), us -> rad/us.
    phi : float
        Laser phase, in rad.
    v_values : numpy.ndarray, shape (n_v,)
        Van der Waals shifts V, in rad/us.
    t_max : float
        Sweep duration, in us.
    numerics : dict
        The ``[numerics]`` table: ``n_steps``, ``sample``, ``check_factor``.
    n_saved_times : int
        Number of evenly spaced output times; (n_saved_times - 1) must divide
        ``n_steps / check_factor``.

    Returns
    -------
    times : numpy.ndarray, shape (n_saved_times,)
        Output times, in us.
    p_00, p_psi_plus : numpy.ndarray, shape (n_v, n_saved_times)
        Populations (dimensionless).
    max_step_change : float
        max over V and output times of |P(dt) - P(check_factor dt)| for both
        populations (dimensionless). For the O(dt^2) midpoint rule with
        check_factor = 2 the error at dt is about a third of this.

    Raises
    ------
    ValueError
        If the output grid does not fit the coarse step grid.
    """
    n_steps = int(numerics["n_steps"])
    factor = int(numerics["check_factor"])
    if n_steps % factor or (n_steps // factor) % (n_saved_times - 1):
        raise ValueError("n_saved_times - 1 must divide n_steps / check_factor.")
    stride = n_steps // (n_saved_times - 1)
    coarse_stride = stride // factor

    p_00 = np.empty((v_values.size, n_saved_times))
    p_psi = np.empty_like(p_00)
    max_change = 0.0
    for i, v in enumerate(v_values):
        times, fine_00, fine_psi = run_sweep(
            omega, detuning, phi, float(v), t_max, n_steps, numerics["sample"]
        )
        _, coarse_00, coarse_psi = run_sweep(
            omega, detuning, phi, float(v), t_max, n_steps // factor, numerics["sample"]
        )
        p_00[i], p_psi[i] = fine_00[::stride], fine_psi[::stride]
        max_change = max(
            max_change,
            float(np.max(np.abs(p_00[i] - coarse_00[::coarse_stride]))),
            float(np.max(np.abs(p_psi[i] - coarse_psi[::coarse_stride]))),
        )
    return times[::stride], p_00, p_psi, max_change


def plot_sweep(
    times: RealArray,
    detuning_mhz: RealArray,
    v_values: RealArray,
    p_00: RealArray,
    p_psi_plus: RealArray,
    plot_cfg: dict[str, Any],
) -> Figure:
    """Plot Delta(t) and the populations in the layout of handbook Fig. 5.2.

    Parameters
    ----------
    times : numpy.ndarray, shape (n_times,)
        Times, in us.
    detuning_mhz : numpy.ndarray, shape (n_times,)
        Detuning Delta / 2pi, in MHz.
    v_values : numpy.ndarray, shape (n_v,)
        Van der Waals shifts V, in rad/us.
    p_00, p_psi_plus : numpy.ndarray, shape (n_v, n_times)
        Populations of |00> and |Psi+> (dimensionless).
    plot_cfg : dict
        The ``[plot]`` table of the config.

    Returns
    -------
    matplotlib.figure.Figure
        The three-panel figure.
    """
    fig, (ax_d, ax_00, ax_psi) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=tuple(plot_cfg["figsize"]),
        gridspec_kw={"height_ratios": plot_cfg["height_ratios"]},
    )
    ax_d.plot(times, detuning_mhz, color=plot_cfg["colour_detuning"])
    d_max = float(np.max(np.abs(detuning_mhz)))
    ax_d.set_ylim(-1.05 * d_max, 1.05 * d_max)
    ax_d.set_ylabel(r"$\Delta/2\pi$ (MHz)")

    v_max = float(np.max(v_values))
    alpha_min = float(plot_cfg["alpha_min"])
    for ax, populations in ((ax_00, p_00), (ax_psi, p_psi_plus)):
        for v, population in zip(v_values, populations, strict=True):
            if v == 0.0:
                ax.plot(
                    times,
                    population,
                    color=plot_cfg["colour_v0"],
                    linewidth=plot_cfg["linewidth_v0"],
                    zorder=3,
                )
            else:
                alpha = alpha_min + (1.0 - alpha_min) * v / v_max
                ax.plot(
                    times,
                    population,
                    color=plot_cfg["colour_v"],
                    alpha=alpha,
                    linewidth=plot_cfg["linewidth_v"],
                    zorder=2,
                )
        ax.set_ylim(0.0, 1.0)

    ax_00.set_ylabel(r"$P_{|00\rangle}$")
    ax_psi.set_ylabel(r"$P_{|\Psi^+\rangle}$")
    ax_psi.set_xlabel(r"Time ($\mu$s)")
    ax_psi.set_xlim(float(times[0]), float(times[-1]))
    fig.align_ylabels()
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
    profile = SWEEP_PROFILES[cfg["sweep"]["profile"]]

    def detuning(t: Any) -> RealArray:
        return profile(t, delta_max, t_max)

    def reversed_detuning(t: Any) -> RealArray:
        return -profile(t, delta_max, t_max)

    v_values = np.asarray(
        mhz_to_angular(
            np.linspace(
                cfg["interaction"]["v_min_mhz"],
                cfg["interaction"]["v_max_mhz"],
                cfg["interaction"]["n_v"],
            )
        ),
        dtype=np.float64,
    )
    numerics = cfg["numerics"]

    times, p_00, p_psi, max_change = compute_populations(
        omega, detuning, phi, v_values, t_max, numerics, int(cfg["output"]["n_saved_times"])
    )

    # Sign check: final P(|Psi+>) with Delta -> -Delta.
    sign_check = []
    for v_mhz in cfg["sign_check"]["v_mhz"]:
        v = mhz_to_angular(float(v_mhz))
        finals = [
            float(
                run_sweep(omega, d, phi, v, t_max, int(numerics["n_steps"]), numerics["sample"])[2][-1]
            )
            for d in (detuning, reversed_detuning)
        ]
        sign_check.append(
            {"v_mhz": float(v_mhz), "p_psi_plus_final": finals[0], "p_psi_plus_final_reversed": finals[1]}
        )

    i_v0 = int(np.argmin(np.abs(v_values)))
    i_vmax = int(np.argmax(v_values))
    summary = {
        "dt_us": t_max / int(numerics["n_steps"]),
        "max_step_doubling_change": max_change,
        "p_psi_plus_final_v_min": float(p_psi[i_v0, -1]),
        "p_00_final_v_min": float(p_00[i_v0, -1]),
        "p_psi_plus_final_v_max": float(p_psi[i_vmax, -1]),
        "p_00_final_v_max": float(p_00[i_vmax, -1]),
        "sign_check": sign_check,
    }

    save_results(
        cfg["output"]["data"],
        {"times_us": times, "v": v_values, "p_00": p_00, "p_psi_plus": p_psi},
        metadata={"config_path": str(args.config), "config": cfg, "summary": summary},
    )

    print(f"dt = {summary['dt_us']:.3g} us ({numerics['sample']}); "
          f"max |P(dt) - P({numerics['check_factor']} dt)| over all V = {max_change:.2e}")
    print(f"V/2pi = {cfg['interaction']['v_min_mhz']} MHz: P_Psi+(t_max) = "
          f"{summary['p_psi_plus_final_v_min']:.6f}, P_00(t_max) = {summary['p_00_final_v_min']:.6f}")
    print(f"V/2pi = {cfg['interaction']['v_max_mhz']} MHz: P_Psi+(t_max) = "
          f"{summary['p_psi_plus_final_v_max']:.6f}, P_00(t_max) = {summary['p_00_final_v_max']:.6f}")
    print("Sign check, final P_Psi+ (Eq. 5.4 as written | Delta -> -Delta):")
    for row in sign_check:
        print(f"  V/2pi = {row['v_mhz']:6.1f} MHz: {row['p_psi_plus_final']:.4f} | "
              f"{row['p_psi_plus_final_reversed']:.4f}")

    use_style()
    detuning_mhz = np.asarray(angular_to_mhz(detuning(times)), dtype=np.float64)
    fig = plot_sweep(times, detuning_mhz, v_values, p_00, p_psi, cfg["plot"])
    pdf_path, _ = save_figure(fig, cfg["output"]["figure"])
    print(f"Wrote {pdf_path} and {cfg['output']['data']}.npz")


if __name__ == "__main__":
    main()
