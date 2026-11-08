"""Propagation for time-dependent Hamiltonians via Trotter-Suzuki stepping.

Advances states with handbook Eq. (5.7),

    |Psi(t + dt)> = exp(-i H(t_s) dt) |Psi(t)>,

where each step applies the *exact* matrix exponential of the Hamiltonian
sampled at a single time t_s in [t, t + dt]. The only approximation is
treating H as constant across a step. Two sampling rules are offered:

``"left"``
    t_s = t, Eq. (5.7) as written. The local error per step is O(dt^2), from
    the term (dt^2 / 2) dH/dt, so the global error at fixed t_max is O(dt).
``"midpoint"``
    t_s = t + dt/2 (the exponential midpoint rule). The first-order error
    cancels by symmetry, leaving a local error O(dt^3) from the commutator
    [H, dH/dt] and d^2H/dt^2, so the global error is O(dt^2). The cost per
    step is identical.

For a time-independent H both rules are exact for any dt. Each step is unitary
to round-off, so the norm is conserved whatever dt is.

Units: hbar = 1, times in us, H / hbar in rad/us.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rydberg.evolution.eigen import HERMITIAN_RTOL, propagator

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

Sample = Literal["left", "midpoint"]

_SAMPLE_OFFSETS: dict[str, float] = {"left": 0.0, "midpoint": 0.5}
"""Position of the sampling time within a step, as a fraction of dt."""

STEP_RTOL = 1e-9
"""Relative tolerance for accepting (t_max - t0) / dt as an integer."""

_CHUNK = 4096
"""Steps whose propagators are built together; bounds memory to a few MB."""


def n_steps_for(t_max: float, dt: float, t0: float = 0.0) -> int:
    """Return the number of steps of size ``dt`` from ``t0`` to ``t_max``.

    Parameters
    ----------
    t_max : float
        Final time, in us. Must exceed ``t0``.
    dt : float
        Step size, in us. Must be positive.
    t0 : float, default 0.0
        Initial time, in us.

    Returns
    -------
    int
        n = (t_max - t0) / dt.

    Raises
    ------
    ValueError
        If ``dt <= 0``, ``t_max <= t0``, or (t_max - t0) / dt is not an
        integer to within a relative tolerance :data:`STEP_RTOL`.
    """
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    duration = t_max - t0
    if not duration > 0:
        raise ValueError(f"t_max ({t_max}) must be greater than t0 ({t0}).")
    ratio = duration / dt
    n = round(ratio)
    if n < 1 or abs(ratio - n) > STEP_RTOL * ratio:
        raise ValueError(
            f"dt = {dt} does not divide t_max - t0 = {duration} into an integer "
            f"number of steps ({ratio}); choose dt = (t_max - t0) / n."
        )
    return n


def _step_propagators(h: ComplexArray, dt: float) -> ComplexArray:
    """Return exp(-i H_k dt) for a stack of Hamiltonians H_k.

    This is :func:`rydberg.evolution.eigen.propagator` applied to many matrices
    at once: U_k = R_k^dagger D_k(dt) R_k with R_k from ``np.linalg.eigh``.

    Parameters
    ----------
    h : numpy.ndarray, shape (m, n, n)
        Hermitian Hamiltonians H_k / hbar, in rad/us.
    dt : float
        Step size, in us.

    Returns
    -------
    numpy.ndarray, shape (m, n, n), complex
        Unitary step propagators (dimensionless).

    Raises
    ------
    ValueError
        If any H_k is not Hermitian.
    """
    scale = np.maximum(np.linalg.norm(h, axis=(-2, -1)), 1.0)
    asymmetry = np.linalg.norm(h - np.conj(np.swapaxes(h, -1, -2)), axis=(-2, -1))
    if np.any(asymmetry > HERMITIAN_RTOL * scale):
        raise ValueError("Hamiltonian is not Hermitian at some time.")
    energies, u = np.linalg.eigh(h)  # H_k = U_k diag(E_k) U_k^dagger, U_k = R_k^dagger
    phases = np.exp(-1j * energies * dt)
    props: ComplexArray = (u * phases[:, None, :]) @ np.conj(np.swapaxes(u, -1, -2))
    return props


def evolve_trotter(
    hamiltonian: Callable[..., ArrayLike],
    psi0: ArrayLike,
    t_max: float,
    dt: float,
    *,
    sample: Sample = "left",
    t0: float = 0.0,
    vectorised: bool = False,
) -> tuple[RealArray, ComplexArray]:
    """Evolve ``psi0`` under a time-dependent Hamiltonian with Eq. (5.7).

    Parameters
    ----------
    hamiltonian : callable
        H(t): maps a time in us to the Hermitian matrix H(t) / hbar in rad/us,
        shape ``(n, n)``. With ``vectorised=True`` it must instead accept a 1-D
        array of times and return shape ``(m, n, n)``, as the callables from
        :func:`rydberg.protocols.detuning_sweep_hamiltonian` do. This is only a
        speed-up; the result is the same.
    psi0 : array_like, shape (n,)
        Initial state |Psi(t0)> (dimensionless, assumed normalised).
    t_max : float
        Final time, in us.
    dt : float
        Step size, in us. (t_max - t0) / dt must be an integer to within a
        relative 1e-9; a non-dividing dt raises rather than silently taking a
        shortened last step, so every step has the same size and the error
        scaling is clean. Use dt = (t_max - t0) / n_steps.
    sample : {"left", "midpoint"}, default "left"
        Where in each step H is evaluated: at its start t (Eq. 5.7 as written,
        global error O(dt)) or at t + dt/2 (global error O(dt^2)).
    t0 : float, default 0.0
        Initial time, in us.
    vectorised : bool, default False
        Whether ``hamiltonian`` accepts an array of times (see above).

    Returns
    -------
    times : numpy.ndarray, shape (n_steps + 1,)
        Times t0, t0 + dt, ..., t_max, in us.
    states : numpy.ndarray, shape (n_steps + 1, n), complex
        |Psi(t)> at each time, starting with ``psi0``, in the trajectory
        layout of :func:`rydberg.evolution.eigen.evolve`.

    Raises
    ------
    ValueError
        If ``dt`` is not positive or does not divide t_max - t0, if
        ``sample`` is unknown, if H(t) is not Hermitian, or if the shapes of
        H(t) and ``psi0`` do not match.

    Notes
    -----
    Steps are processed in chunks: the sampled Hamiltonians of a chunk are
    diagonalised together, and the resulting step propagators are then applied
    to the state in order. Each step propagator is exp(-i H(t_s) dt) to
    round-off, exactly as from :func:`rydberg.evolution.eigen.propagator`.
    """
    if sample not in _SAMPLE_OFFSETS:
        raise ValueError(
            f"sample must be one of {sorted(_SAMPLE_OFFSETS)}, got {sample!r}."
        )
    n_steps = n_steps_for(t_max, dt, t0)
    psi = np.asarray(psi0, dtype=np.complex128)
    if psi.ndim != 1:
        raise ValueError(f"psi0 must be a single state vector, got shape {psi.shape}.")

    times = t0 + dt * np.arange(n_steps + 1, dtype=np.float64)
    times[-1] = t_max  # remove accumulated round-off in the last time
    sample_times = t0 + dt * (np.arange(n_steps) + _SAMPLE_OFFSETS[sample])

    states = np.empty((n_steps + 1, psi.size), dtype=np.complex128)
    states[0] = psi
    for start in range(0, n_steps, _CHUNK):
        chunk = sample_times[start : start + _CHUNK]
        if vectorised:
            h = np.asarray(hamiltonian(chunk), dtype=np.complex128)
        else:
            h = np.stack(
                [np.asarray(hamiltonian(float(t)), dtype=np.complex128) for t in chunk]
            )
        if h.shape != (chunk.size, psi.size, psi.size):
            raise ValueError(
                f"Hamiltonian of shape {h.shape[1:]} is incompatible with a state "
                f"of dimension {psi.size}."
            )
        for k, u in enumerate(_step_propagators(h, dt), start=start):
            psi = u @ psi
            states[k + 1] = psi
    return times, states


def evolve_trotter_reference(
    hamiltonian: Callable[[float], ArrayLike],
    psi0: ArrayLike,
    t_max: float,
    dt: float,
    *,
    sample: Sample = "left",
    t0: float = 0.0,
) -> tuple[RealArray, ComplexArray]:
    """Evolve with Eq. (5.7) using one :func:`eigen.propagator` call per step.

    A deliberately literal, unbatched implementation of :func:`evolve_trotter`,
    kept as an independent check that the batched version is identical. It is
    about ten times slower.

    Parameters
    ----------
    hamiltonian : callable
        H(t): maps a time in us to H(t) / hbar in rad/us, shape ``(n, n)``.
    psi0 : array_like, shape (n,)
        Initial state |Psi(t0)> (dimensionless).
    t_max : float
        Final time, in us.
    dt : float
        Step size, in us; must divide t_max - t0.
    sample : {"left", "midpoint"}, default "left"
        Sampling time within each step, as in :func:`evolve_trotter`.
    t0 : float, default 0.0
        Initial time, in us.

    Returns
    -------
    times : numpy.ndarray, shape (n_steps + 1,)
        Times, in us.
    states : numpy.ndarray, shape (n_steps + 1, n), complex
        |Psi(t)> at each time.

    Raises
    ------
    ValueError
        As for :func:`evolve_trotter`.
    """
    if sample not in _SAMPLE_OFFSETS:
        raise ValueError(
            f"sample must be one of {sorted(_SAMPLE_OFFSETS)}, got {sample!r}."
        )
    n_steps = n_steps_for(t_max, dt, t0)
    psi = np.asarray(psi0, dtype=np.complex128)
    times = t0 + dt * np.arange(n_steps + 1, dtype=np.float64)
    times[-1] = t_max
    states = np.empty((n_steps + 1, psi.size), dtype=np.complex128)
    states[0] = psi
    for k in range(n_steps):
        t_s = t0 + dt * (k + _SAMPLE_OFFSETS[sample])
        psi = propagator(hamiltonian(t_s), dt) @ psi
        states[k + 1] = psi
    return times, states
