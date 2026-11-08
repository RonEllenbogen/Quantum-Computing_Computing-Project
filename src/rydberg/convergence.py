"""Error metrics and numerical convergence studies.

Power-law fits of error against step size, comparison of trajectories on a
common time grid, a precise significant-figures criterion, and searches for the
largest step size meeting a tolerance.

Significant figures
-------------------
A value P agrees with a reference P_ref to ``n`` significant figures here when

    |P - P_ref| <= 0.5 * 10^(-n) * |P_ref|,

i.e. for n = 4, |P - P_ref| <= 5e-5 |P_ref|. This is half a unit in the n-th
significant figure when the leading digit of P_ref is close to 9 (e.g. for a
population P_ref ~ 0.99), and stricter otherwise, so it *guarantees* n correct
significant figures whatever the leading digit. (The half-unit itself ranges
from 0.5 * 10^(-n) to 0.5 * 10^(1-n) relative to |P_ref|.)

For whole trajectories, where populations pass through zero and a relative
criterion is meaningless, the analogous absolute tolerance 0.5 * 10^(-n) is
used on the maximum absolute population error. Populations lie in [0, 1], so
for n = 4 this is half a unit in the fourth decimal place.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

RealArray = NDArray[np.float64]

GRID_RTOL = 1e-9
"""Relative tolerance (in units of the fine step) for matching time grids."""


def fit_power_law(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Fit y = A x^p by linear least squares in log space.

    Fits log y = log A + p log x.

    Parameters
    ----------
    x : array_like, shape (m,)
        Positive abscissae, e.g. step sizes dt in us.
    y : array_like, shape (m,)
        Positive ordinates, e.g. errors (dimensionless).

    Returns
    -------
    p : float
        Fitted exponent (e.g. the convergence order).
    a : float
        Fitted prefactor A, in units of y / x^p.

    Raises
    ------
    ValueError
        If the shapes differ, fewer than two points are given, or any value is
        not strictly positive and finite.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError(f"x and y must be 1-D of equal length, got {xs.shape}, {ys.shape}.")
    if xs.size < 2:
        raise ValueError("At least two points are needed for a power-law fit.")
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise ValueError("x and y must be finite.")
    if np.any(xs <= 0) or np.any(ys <= 0):
        raise ValueError("x and y must be strictly positive for a log-log fit.")
    p, log_a = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(p), float(np.exp(log_a))


def sig_fig_tolerance(reference: float, n_sig: int = 4) -> float:
    """Return the absolute tolerance 0.5 * 10^(-n_sig) * |reference|.

    See the module docstring for why this guarantees ``n_sig`` significant
    figures.

    Parameters
    ----------
    reference : float
        Reference value (any units).
    n_sig : int, default 4
        Number of significant figures required.

    Returns
    -------
    float
        Tolerance on |value - reference|, in the units of ``reference``.

    Raises
    ------
    ValueError
        If ``n_sig < 1``.
    """
    if n_sig < 1:
        raise ValueError(f"n_sig must be at least 1, got {n_sig}.")
    return 0.5 * 10.0 ** (-n_sig) * abs(reference)


def agrees_to_sig_figs(value: float, reference: float, n_sig: int = 4) -> bool:
    """Return whether |value - reference| <= 0.5 * 10^(-n_sig) * |reference|.

    Parameters
    ----------
    value : float
        Approximate value (any units).
    reference : float
        Reference value, in the same units.
    n_sig : int, default 4
        Number of significant figures required.

    Returns
    -------
    bool
        True if ``value`` agrees with ``reference`` to ``n_sig`` significant
        figures in the sense of the module docstring.
    """
    return bool(abs(value - reference) <= sig_fig_tolerance(reference, n_sig))


def max_abs_error(values: ArrayLike, reference: ArrayLike) -> float:
    """Return max |values - reference| over all elements.

    Parameters
    ----------
    values, reference : array_like
        Arrays of equal shape, e.g. population trajectories (dimensionless).

    Returns
    -------
    float
        Maximum absolute difference, in the units of the inputs.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    a = np.asarray(values)
    b = np.asarray(reference)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")
    return float(np.max(np.abs(a - b)))


def subsample_to_grid(
    fine_times: ArrayLike, fine_values: ArrayLike, coarse_times: ArrayLike
) -> NDArray[np.generic]:
    """Return the rows of ``fine_values`` at the times ``coarse_times``.

    Used to compare a coarse trajectory with a fine reference on a common
    grid without interpolation, which would add its own error. Every coarse
    time must coincide with a fine time.

    Parameters
    ----------
    fine_times : array_like, shape (n_fine,)
        Uniformly spaced times of the fine trajectory, in us.
    fine_values : array_like, shape (n_fine, ...)
        Values on the fine grid (e.g. populations or states).
    coarse_times : array_like, shape (n_coarse,)
        Times at which values are wanted, in us.

    Returns
    -------
    numpy.ndarray, shape (n_coarse, ...)
        ``fine_values`` at the coarse times.

    Raises
    ------
    ValueError
        If any coarse time does not coincide with a fine time to within
        :data:`GRID_RTOL` of the fine step.
    """
    tf = np.asarray(fine_times, dtype=np.float64)
    vf = np.asarray(fine_values)
    tc = np.asarray(coarse_times, dtype=np.float64)
    if tf.size < 2 or vf.shape[0] != tf.size:
        raise ValueError("fine_times must have >= 2 entries matching fine_values.")
    step = (tf[-1] - tf[0]) / (tf.size - 1)
    index = np.rint((tc - tf[0]) / step).astype(np.int64)
    if np.any(index < 0) or np.any(index >= tf.size):
        raise ValueError("Coarse times lie outside the fine grid.")
    if np.any(np.abs(tf[index] - tc) > GRID_RTOL * step + 1e-15 * np.abs(tc)):
        raise ValueError("Coarse times do not coincide with points of the fine grid.")
    result: NDArray[np.generic] = vf[index]
    return result


def largest_passing_step(
    steps: ArrayLike, errors: ArrayLike, tolerance: float
) -> float | None:
    """Return the largest step whose error, and that of every smaller step, passes.

    Requiring all smaller steps to pass as well guards against an error that
    passes by accident at a coarse step (e.g. an oscillating error crossing
    zero) while failing at finer ones.

    Parameters
    ----------
    steps : array_like, shape (m,)
        Step sizes, in any order (e.g. dt in us).
    errors : array_like, shape (m,)
        Error at each step size (non-negative).
    tolerance : float
        Maximum acceptable error, in the units of ``errors``.

    Returns
    -------
    float or None
        The largest qualifying step, or None if even the smallest step fails.

    Raises
    ------
    ValueError
        If the shapes differ or the input is empty.
    """
    h = np.asarray(steps, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    if h.shape != e.shape or h.ndim != 1 or h.size == 0:
        raise ValueError("steps and errors must be non-empty 1-D arrays of equal length.")
    order = np.argsort(h)
    passing = np.cumprod(e[order] <= tolerance).astype(bool)  # all smaller pass
    if not passing[0]:
        return None
    return float(h[order][np.flatnonzero(passing)[-1]])


def min_passing_steps(
    error_of_steps: Callable[[int], float],
    n_fail: int,
    n_pass: int,
    tolerance: float,
) -> int:
    """Bisect for the smallest step count n whose error meets ``tolerance``.

    Assumes the error is monotonically non-increasing in n between the two
    brackets, which holds in the asymptotic regime of a convergent method.
    The largest acceptable step is then (t_max - t0) / n.

    Parameters
    ----------
    error_of_steps : callable
        Maps a number of steps n to the error of a run with n steps.
    n_fail : int
        A step count known to fail (error > tolerance).
    n_pass : int
        A larger step count known to pass (error <= tolerance).
    tolerance : float
        Maximum acceptable error, in the units returned by ``error_of_steps``.

    Returns
    -------
    int
        The smallest n in (n_fail, n_pass] with error <= tolerance.

    Raises
    ------
    ValueError
        If ``n_fail >= n_pass``, or the brackets do not fail and pass.
    """
    if n_fail >= n_pass:
        raise ValueError(f"Need n_fail < n_pass, got {n_fail} and {n_pass}.")
    if error_of_steps(n_fail) <= tolerance:
        raise ValueError(f"n_fail = {n_fail} already meets the tolerance.")
    if error_of_steps(n_pass) > tolerance:
        raise ValueError(f"n_pass = {n_pass} does not meet the tolerance.")
    lo, hi = n_fail, n_pass
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if error_of_steps(mid) <= tolerance:
            hi = mid
        else:
            lo = mid
    return hi
