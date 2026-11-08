"""Find the largest Trotter time step giving populations to 4 significant figures.

The cubic detuning sweep of handbook Fig. 5.2 is propagated with Eq. (5.7) using
left-point sampling (H at the start of each step, as written in the handbook)
and midpoint sampling (H at the middle of each step), for step sizes
dt = t_max / n. Errors are measured against a midpoint run with a much smaller
step, which is itself cross-checked against scipy's adaptive DOP853 integrator.

Two error measures are used (see :mod:`rydberg.convergence`):

trajectory (the headline criterion)
    max over the coarse time grid and over {P_00, P_Psi+} of the absolute
    population error, required to be <= 0.5e-4. Since P_Psi- stays zero by
    exchange symmetry, the error in P_rr is then at most twice this.
final (a weak test, reported for comparison)
    |P_Psi+(t_max; dt) - P_ref|, required to be <= 0.5e-4 |P_ref|. Because
    P_Psi+(t_max) is close to its maximum of 1, it is only quadratically
    sensitive to errors in the state: at the left-rule dt that passes this
    test, the small population P_00(t_max) can still be wrong at the percent
    level. Use the trajectory criterion (or 1 - P) to judge accuracy.

The largest acceptable dt is the largest scanned dt for which that dt and every
smaller scanned dt pass, refined by bisection on the number of steps.

Usage
-----
    uv run python scripts/timestep_convergence.py --config configs/timestep_convergence.toml
"""

import argparse
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from rydberg.config import load_config
from rydberg.convergence import (
    fit_power_law,
    largest_passing_step,
    max_abs_error,
    min_passing_steps,
    sig_fig_tolerance,
    subsample_to_grid,
)
from rydberg.evolution.trotter import evolve_trotter
from rydberg.io import save_results
from rydberg.plotting import save_figure, use_style
from rydberg.protocols import SWEEP_PROFILES, TimeDependentHamiltonian, detuning_sweep_hamiltonian
from rydberg.states import KET_00, PSI_PLUS, overlap_mod_squared
from rydberg.units import mhz_to_angular

DEFAULT_CONFIG = Path("configs/timestep_convergence.toml")

RealArray = NDArray[np.float64]
METRICS = ("trajectory", "final")


def populations(states: NDArray[np.complex128]) -> RealArray:
    """Return [P(|00>), P(|Psi+>)] along a trajectory, shape (n_times, 2)."""
    return np.stack(
        [overlap_mod_squared(states, KET_00), overlap_mod_squared(states, PSI_PLUS)],
        axis=-1,
    ).astype(np.float64)


def run(
    h: TimeDependentHamiltonian, t_max: float, n_steps: int, sample: str
) -> tuple[RealArray, RealArray]:
    """Return times (us) and populations for ``n_steps`` Trotter steps from |00>."""
    times, states = evolve_trotter(
        h, KET_00, t_max, t_max / n_steps, sample=sample, vectorised=True  # type: ignore[arg-type]
    )
    return times, populations(states)


def divisors_in_range(n: int, lo: int, hi: int) -> NDArray[np.int64]:
    """Return the divisors of ``n`` in [lo, hi], ascending."""
    small = [d for d in range(1, math.isqrt(n) + 1) if n % d == 0]
    all_divisors = sorted(set(small + [n // d for d in small]))
    return np.array([d for d in all_divisors if lo <= d <= hi], dtype=np.int64)


class ErrorModel:
    """Errors of a Trotter run with n steps against fine midpoint references.

    Parameters
    ----------
    h : callable
        Vectorised H(t), in rad/us.
    t_max : float
        Sweep duration, in us.
    sample : str
        Sampling rule of the runs being tested.
    reference : tuple of numpy.ndarray
        Times (us) and populations of the main reference run.
    refine_min_steps : int
        Minimum step count of the on-grid references built for step counts
        that do not divide the main reference.
    """

    def __init__(
        self,
        h: TimeDependentHamiltonian,
        t_max: float,
        sample: str,
        reference: tuple[RealArray, RealArray],
        refine_min_steps: int,
    ) -> None:
        self.h, self.t_max, self.sample = h, t_max, sample
        self.ref_times, self.ref_pops = reference
        self.n_ref = self.ref_times.size - 1
        self.refine_min_steps = refine_min_steps
        self._runs: dict[int, tuple[RealArray, RealArray]] = {}
        self._cache: dict[tuple[int, str], float] = {}

    def _run(self, n: int) -> tuple[RealArray, RealArray]:
        """Return (times, populations) of the tested method with ``n`` steps."""
        if n not in self._runs:
            self._runs.clear()  # keep at most one trajectory in memory
            self._runs[n] = run(self.h, self.t_max, n, self.sample)
        return self._runs[n]

    def error(self, n: int, metric: str) -> float:
        """Return the named error ("final" or "trajectory") for ``n`` steps.

        Each metric is computed only when asked for, because the trajectory
        error at a step count that does not divide the main reference needs a
        new on-grid reference run.
        """
        key = (n, metric)
        if key not in self._cache:
            times, pops = self._run(n)
            if metric == "final":
                value = abs(pops[-1, 1] - self.ref_pops[-1, 1])
            elif self.n_ref % n == 0:
                value = max_abs_error(
                    pops, subsample_to_grid(self.ref_times, self.ref_pops, times)
                )
            else:
                m = math.ceil(self.refine_min_steps / n)
                ref_times, ref_pops = run(self.h, self.t_max, n * m, "midpoint")
                value = max_abs_error(pops, subsample_to_grid(ref_times, ref_pops, times))
            self._cache[key] = float(value)
        return self._cache[key]

    def metric(self, name: str) -> Callable[[int], float]:
        """Return n -> error for the named metric ("final" or "trajectory")."""
        return lambda n: self.error(n, name)


def largest_step(
    model: ErrorModel,
    metric: str,
    steps: NDArray[np.int64],
    errors: RealArray,
    tolerance: float,
    numerics: dict[str, Any],
) -> dict[str, Any]:
    """Return the largest acceptable dt for one method and metric.

    Parameters
    ----------
    model : ErrorModel
        Error evaluator for the method.
    metric : str
        "final" or "trajectory".
    steps : numpy.ndarray
        Scanned step counts, ascending.
    errors : numpy.ndarray
        Errors at the scanned step counts.
    tolerance : float
        Acceptance threshold (dimensionless).
    numerics : dict
        The ``[numerics]`` table.

    Returns
    -------
    dict
        ``n_steps`` and ``dt_us`` of the result (None if no scanned dt
        passes), the scan bracket, and for the final metric whether a dense
        check above the result passed.
    """
    t_max = model.t_max
    dt_pass = largest_passing_step(t_max / steps, errors, tolerance)
    if dt_pass is None:
        return {"n_steps": None, "dt_us": None}
    n_pass = int(round(t_max / dt_pass))
    smaller = steps[steps < n_pass]
    result: dict[str, Any] = {"bracket_n": [int(smaller[-1]) if smaller.size else None, n_pass]}
    n_star = n_pass
    if numerics["refine"] and smaller.size:
        n_star = min_passing_steps(model.metric(metric), int(smaller[-1]), n_pass, tolerance)
    if metric == "final":
        # The final-value error is not monotonic at coarse dt; check densely above.
        upper = min(n_star + int(numerics["n_verify"]), n_pass)
        failing = [n for n in range(n_star, upper + 1) if model.metric(metric)(n) > tolerance]
        result["dense_check_range_n"] = [n_star, upper]
        result["dense_check_failures_n"] = failing
        if failing:
            n_star = max(failing) + 1
    result["n_steps"] = n_star
    result["dt_us"] = t_max / n_star
    result["error_at_result"] = model.metric(metric)(n_star)
    return result


def study_case(
    case: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Run the full convergence study for one (profile, V) case.

    Parameters
    ----------
    case : dict
        One entry of ``[[cases]]``.
    cfg : dict
        The full config.

    Returns
    -------
    dict
        Scanned dt, errors per method and metric, fitted orders, tolerances,
        largest acceptable dt and the solve_ivp cross-check.
    """
    numerics = cfg["numerics"]
    omega = mhz_to_angular(float(cfg["drive"]["omega_mhz"]))
    phi = float(cfg["drive"]["phi"])
    delta_max = mhz_to_angular(float(cfg["sweep"]["delta_max_mhz"]))
    t_max = float(cfg["sweep"]["t_max_us"])
    v = mhz_to_angular(float(case["v_mhz"]))
    shape = SWEEP_PROFILES[case["profile"]]
    h = detuning_sweep_hamiltonian(omega, lambda t: shape(t, delta_max, t_max), phi, v)

    n_ref = int(numerics["reference_steps"])
    reference = run(h, t_max, n_ref, "midpoint")
    p_ref_final = float(reference[1][-1, 1])

    # Independent check of the reference with an adaptive Runge-Kutta solver.
    val = cfg["validation"]
    stride = n_ref // (int(val["n_eval"]) - 1)
    t_eval = reference[0][::stride]
    sol = solve_ivp(
        lambda t, y: -1j * (h(t) @ y),
        (0.0, t_max),
        KET_00.astype(np.complex128),
        method=val["method"],
        t_eval=t_eval,
        rtol=float(val["rtol"]),
        atol=float(val["atol"]),
    )
    ivp_pops = populations(sol.y.T)
    ivp = {
        "final_p_psi_plus_difference": float(abs(ivp_pops[-1, 1] - p_ref_final)),
        "max_population_difference": max_abs_error(ivp_pops, reference[1][::stride]),
        "n_function_evaluations": int(sol.nfev),
    }

    n_sig = int(numerics["n_sig"])
    tolerances = {"final": sig_fig_tolerance(p_ref_final, n_sig), "trajectory": 0.5 * 10.0**-n_sig}
    steps = divisors_in_range(n_ref, int(numerics["n_min"]), int(numerics["n_max"]))
    dts = t_max / steps

    out: dict[str, Any] = {
        "label": case["label"], "p_ref_final": p_ref_final, "tolerances": tolerances,
        "solve_ivp": ivp, "dt_us": dts, "errors": {}, "orders": {}, "largest_dt": {},
    }
    # Fit windows per metric: the final-value error of the left rule only
    # reaches its asymptotic regime at smaller dt than the trajectory error.
    windows = {
        m: (dts >= float(numerics["fit_dt_min_us"][m])) & (dts <= float(numerics["fit_dt_max_us"][m]))
        for m in METRICS
    }
    for method in numerics["methods"]:
        model = ErrorModel(h, t_max, method, reference, int(numerics["refine_reference_min_steps"]))
        errs = np.array([[model.error(int(n), m) for m in METRICS] for n in steps])
        out["errors"][method] = {m: errs[:, k] for k, m in enumerate(METRICS)}
        out["orders"][method] = {}
        out["largest_dt"][method] = {}
        for k, metric in enumerate(METRICS):
            use = windows[metric] & (errs[:, k] > float(numerics["fit_error_floor"]))
            if np.count_nonzero(use) >= 3:
                p, a = fit_power_law(dts[use], errs[use, k])
                rms = float(np.sqrt(np.mean(
                    (np.log(errs[use, k]) - np.log(a * dts[use] ** p)) ** 2
                )))
                out["orders"][method][metric] = {"order": p, "prefactor": a, "rms_log_residual": rms}
            out["largest_dt"][method][metric] = largest_step(
                model, metric, steps, errs[:, k], tolerances[metric], numerics
            )
    return out


def plot_convergence(result: dict[str, Any], cfg: dict[str, Any]) -> Figure:
    """Plot error against dt for both metrics of the main case.

    Parameters
    ----------
    result : dict
        Output of :func:`study_case` for the main case.
    cfg : dict
        The full config.

    Returns
    -------
    matplotlib.figure.Figure
        Two log-log panels: max trajectory error (the headline criterion) and
        the final P(|Psi+>) error (a weak test, since P is close to 1).
    """
    plot_cfg = cfg["plot"]
    fig, axes = plt.subplots(1, 2, figsize=tuple(plot_cfg["figsize"]), sharex=True)
    dts = result["dt_us"]
    titles = {
        "trajectory": r"(a) all populations, all times: $\mathrm{max}_t\,|P(t) - P_{\rm ref}(t)|$",
        "final": r"(b) final value only: $|P_{|\Psi^+\rangle}(t_{\max}) - P_{\rm ref}|$",
    }
    for ax, metric in zip(axes, METRICS, strict=True):
        for method in cfg["numerics"]["methods"]:
            c = plot_cfg["colours"][method]
            errs = result["errors"][method][metric]
            fit = result["orders"][method].get(metric)
            label = method
            if fit is not None:
                label += f", slope {fit['order']:.2f}"
                if fit["rms_log_residual"] < 0.3:
                    lo = cfg["numerics"]["fit_dt_min_us"][metric]
                    hi = cfg["numerics"]["fit_dt_max_us"][metric]
                    xs = np.array([lo, hi])
                    ax.loglog(xs, fit["prefactor"] * xs ** fit["order"], color=c,
                              linewidth=0.8, alpha=0.6)
                else:
                    label += " (irregular)"
            best = result["largest_dt"][method][metric]
            if best["dt_us"] is not None:
                ax.axvline(best["dt_us"], color=c, linestyle=":", linewidth=1.0)
                mantissa, exponent = f"{best['dt_us']:.2e}".split("e")
                label += "\n" + (
                    rf"largest $\delta t = {mantissa}\times10^{{{int(exponent)}}}$ $\mu$s"
                )
            ax.loglog(dts, errs, linestyle="none", marker=plot_cfg["markers"][method],
                      markersize=3, color=c, label=label)
        tol = result["tolerances"][metric]
        ax.axhline(tol, color=plot_cfg["colour_threshold"], linestyle="--", linewidth=0.8,
                   label="4 s.f. threshold")
        ax.set_title(titles[metric], loc="left")
        ax.set_xlabel(r"Time step $\delta t$ ($\mu$s)")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Population error")
    return fig


def _jsonable(result: dict[str, Any]) -> dict[str, Any]:
    """Return the scalar parts of a case result (arrays go in the .npz file)."""
    return {k: result[k] for k in ("label", "p_ref_final", "tolerances", "solve_ivp",
                                   "orders", "largest_dt")}


def main(argv: Sequence[str] | None = None) -> None:
    """Run the study for every case, save the data and write the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    results = [study_case(case, cfg) for case in cfg["cases"]]

    arrays: dict[str, Any] = {}
    for i, res in enumerate(results):
        arrays[f"case{i}_dt_us"] = res["dt_us"]
        for method, by_metric in res["errors"].items():
            for metric, errs in by_metric.items():
                arrays[f"case{i}_{method}_{metric}_error"] = errs
    save_results(
        cfg["output"]["data"],
        arrays,
        metadata={"config_path": str(args.config), "config": cfg,
                  "cases": [_jsonable(r) for r in results]},
    )

    for res in results:
        ivp = res["solve_ivp"]
        print(f"=== {res['label']} ===")
        print(f"  reference P_Psi+(t_max) = {res['p_ref_final']:.10f}")
        print(f"  reference vs solve_ivp: final P_Psi+ diff = {ivp['final_p_psi_plus_difference']:.2e}, "
              f"max population diff = {ivp['max_population_difference']:.2e}")
        for method in cfg["numerics"]["methods"]:
            for metric in METRICS:
                fit = res["orders"][method].get(metric)
                best = res["largest_dt"][method][metric]
                order = (f"order {fit['order']:.3f} (rms log resid {fit['rms_log_residual']:.2f})"
                         if fit else "order n/a")
                dt = (f"largest dt = {best['dt_us']:.4g} us (n = {best['n_steps']}, "
                      f"error {best['error_at_result']:.2e} <= {res['tolerances'][metric]:.2e})"
                      if best["dt_us"] is not None else "no scanned dt passes")
                print(f"  {method:8s} {metric:10s}: {order}; {dt}")
                if best.get("dense_check_failures_n"):
                    print(f"    dense check moved result; failures at n = {best['dense_check_failures_n']}")

    use_style()
    fig = plot_convergence(results[0], cfg)
    pdf_path, _ = save_figure(fig, cfg["output"]["figure"])
    print(f"Wrote {pdf_path} and {cfg['output']['data']}.npz")


if __name__ == "__main__":
    main()
