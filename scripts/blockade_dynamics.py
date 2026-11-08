"""Reproduce the Rydberg-blockade dynamics for a range of interaction strengths.

Two atoms start in |00> and are driven resonantly with Rabi frequency Omega.
For each van der Waals shift V the time-independent Hamiltonian of handbook
Eq. (5.4) is propagated exactly by eigendecomposition, and the populations of
|00> and of the Bell state |Psi+> = (|0r> + |r0>) / sqrt(2) are recorded.

The figure reproduces handbook Fig. 5.1: V = 0 in blue, V > 0 in red with
opacity increasing with V. For V >> Omega the curves converge on the blockade
limit, a two-level oscillation between |00> and |Psi+> at sqrt(2) Omega.

Usage
-----
    uv run python scripts/blockade_dynamics.py --config configs/blockade_dynamics.toml
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
from rydberg.evolution.eigen import evolve
from rydberg.hamiltonians import two_atom_hamiltonian
from rydberg.io import save_results
from rydberg.plotting import save_figure, use_style
from rydberg.states import KET_00, PSI_PLUS, overlap_mod_squared
from rydberg.units import mhz_to_angular

DEFAULT_CONFIG = Path("configs/blockade_dynamics.toml")

RealArray = NDArray[np.float64]


def compute_populations(
    omega: float,
    delta: float,
    phi: float,
    v_values: RealArray,
    times: RealArray,
) -> tuple[RealArray, RealArray]:
    """Return P(|00>) and P(|Psi+>) for each interaction strength.

    Parameters
    ----------
    omega : float
        Rabi frequency Omega, in rad/us.
    delta : float
        Detuning Delta, in rad/us.
    phi : float
        Laser phase, in rad.
    v_values : numpy.ndarray, shape (n_v,)
        Van der Waals shifts V, in rad/us.
    times : numpy.ndarray, shape (n_times,)
        Evolution times, in us.

    Returns
    -------
    p_00, p_psi_plus : numpy.ndarray, shape (n_v, n_times)
        Populations of |00> and |Psi+> (dimensionless), starting from |00>.
    """
    p_00 = np.empty((v_values.size, times.size))
    p_psi_plus = np.empty_like(p_00)
    for i, v in enumerate(v_values):
        psi_t = evolve(two_atom_hamiltonian(omega, delta, phi, float(v)), KET_00, times)
        p_00[i] = overlap_mod_squared(psi_t, KET_00)
        p_psi_plus[i] = overlap_mod_squared(psi_t, PSI_PLUS)
    return p_00, p_psi_plus


def plot_populations(
    times: RealArray,
    v_values: RealArray,
    p_00: RealArray,
    p_psi_plus: RealArray,
    plot_cfg: dict[str, Any],
) -> Figure:
    """Plot the populations in the layout of handbook Fig. 5.1.

    Parameters
    ----------
    times : numpy.ndarray, shape (n_times,)
        Evolution times, in us.
    v_values : numpy.ndarray, shape (n_v,)
        Van der Waals shifts V, in rad/us. Curves with V = 0 are drawn in the
        reference colour, all others in red with opacity rising linearly in V.
    p_00, p_psi_plus : numpy.ndarray, shape (n_v, n_times)
        Populations of |00> and |Psi+> (dimensionless).
    plot_cfg : dict
        The ``[plot]`` table of the config.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    fig, (ax_00, ax_psi) = plt.subplots(
        2, 1, sharex=True, figsize=tuple(plot_cfg["figsize"])
    )
    v_max = float(np.max(v_values))
    alpha_min = float(plot_cfg["alpha_min"])

    for ax, populations in ((ax_00, p_00), (ax_psi, p_psi_plus)):
        for v, population in zip(v_values, populations, strict=True):
            if v == 0.0:
                # Drawn on top of the red curves, as in Fig. 5.1.
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
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    """Run the computation, save the data and write the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    omega = mhz_to_angular(float(cfg["drive"]["omega_mhz"]))
    delta = mhz_to_angular(float(cfg["drive"]["delta_mhz"]))
    phi = float(cfg["drive"]["phi"])
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
    times = np.linspace(0.0, cfg["time"]["t_max_us"], cfg["time"]["n_times"])

    p_00, p_psi_plus = compute_populations(omega, delta, phi, v_values, times)

    save_results(
        cfg["output"]["data"],
        {"times_us": times, "v": v_values, "p_00": p_00, "p_psi_plus": p_psi_plus},
        metadata={"config_path": str(args.config), "config": cfg},
    )

    use_style()
    fig = plot_populations(times, v_values, p_00, p_psi_plus, cfg["plot"])
    pdf_path, _ = save_figure(fig, cfg["output"]["figure"])
    print(f"Wrote {pdf_path} and {cfg['output']['data']}.npz")


if __name__ == "__main__":
    main()
