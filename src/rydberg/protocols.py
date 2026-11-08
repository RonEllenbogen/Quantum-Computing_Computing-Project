"""Time-dependent control profiles for the laser parameters.

Detuning sweeps Delta(t) through resonance, and a helper that turns a sweep into
a time-dependent two-atom Hamiltonian H(t) for :mod:`rydberg.evolution.trotter`.

Units: hbar = 1, times in us, Omega, Delta and V in rad/us, phi in rad.

Both sweeps run from -Delta_max at t = 0 to +Delta_max at t = t_max, cross
resonance (Delta = 0) at t_max / 2 and are odd about that point. The cubic
sweep of handbook Fig. 5.2 has zero slope at resonance, so it spends longer
near Delta = 0, where the gap of H(t) is smallest, than the linear sweep does.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rydberg.hamiltonians import two_atom_hamiltonian

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

DetuningProfile = Callable[[ArrayLike], RealArray]
"""A detuning profile Delta(t): time(s) in us to detuning(s) in rad/us."""

TimeDependentHamiltonian = Callable[[ArrayLike], ComplexArray]
"""H(t): time(s) in us to H / hbar in rad/us, with shape ``t.shape + (n, n)``."""


def _reduced_time(t: ArrayLike, t_max: float) -> RealArray:
    """Return s = (t - t_max/2) / t_max, which runs from -1/2 to 1/2.

    Parameters
    ----------
    t : float or array_like
        Time(s), in us.
    t_max : float
        Sweep duration, in us. Must be positive.

    Returns
    -------
    numpy.ndarray
        Dimensionless reduced time, same shape as ``t``.

    Raises
    ------
    ValueError
        If ``t_max`` is not positive.
    """
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}.")
    s: RealArray = (np.asarray(t, dtype=np.float64) - t_max / 2) / t_max
    return s


def cubic_detuning(t: ArrayLike, delta_max: float, t_max: float) -> RealArray:
    """Return the cubic sweep Delta(t) = 8 Delta_max [(t - t_max/2) / t_max]^3.

    This is the profile of handbook Fig. 5.2. It is flat at resonance:
    dDelta/dt = 0 at t = t_max / 2.

    Parameters
    ----------
    t : float or array_like
        Time(s), in us. Values outside [0, t_max] follow the same polynomial.
    delta_max : float
        Detuning amplitude Delta_max, in rad/us. Delta(0) = -Delta_max and
        Delta(t_max) = +Delta_max.
    t_max : float
        Sweep duration, in us.

    Returns
    -------
    numpy.ndarray
        Detuning Delta(t), in rad/us, same shape as ``t`` (0-d for scalar t).

    Raises
    ------
    ValueError
        If ``t_max`` is not positive.
    """
    delta: RealArray = 8.0 * delta_max * _reduced_time(t, t_max) ** 3
    return delta


def linear_detuning(t: ArrayLike, delta_max: float, t_max: float) -> RealArray:
    """Return the linear sweep Delta(t) = 2 Delta_max (t - t_max/2) / t_max.

    It has the same endpoints and resonance time as :func:`cubic_detuning`,
    with constant sweep rate dDelta/dt = 2 Delta_max / t_max.

    Parameters
    ----------
    t : float or array_like
        Time(s), in us.
    delta_max : float
        Detuning amplitude Delta_max, in rad/us.
    t_max : float
        Sweep duration, in us.

    Returns
    -------
    numpy.ndarray
        Detuning Delta(t), in rad/us, same shape as ``t`` (0-d for scalar t).

    Raises
    ------
    ValueError
        If ``t_max`` is not positive.
    """
    delta: RealArray = 2.0 * delta_max * _reduced_time(t, t_max)
    return delta


SWEEP_PROFILES: dict[str, Callable[[ArrayLike, float, float], RealArray]] = {
    "cubic": cubic_detuning,
    "linear": linear_detuning,
}
"""Sweep profiles by name, for selecting them from a config file."""


def detuning_sweep_hamiltonian(
    omega: float,
    detuning: DetuningProfile,
    phi: float = 0.0,
    v: float = 0.0,
) -> TimeDependentHamiltonian:
    """Return H(t) = two-atom Hamiltonian with Delta -> Delta(t).

    The returned callable gives exactly
    ``two_atom_hamiltonian(omega, detuning(t), phi, v)`` at each time, but
    accepts an array of times and builds every matrix at once. This uses the
    fact that Eq. (5.4) is linear in Delta,

        H(t) = H(Delta = 0) + Delta(t) dH/dDelta,

    with both terms taken from :func:`rydberg.hamiltonians.two_atom_hamiltonian`,
    so no sign convention is restated here.

    Parameters
    ----------
    omega : float
        Rabi frequency Omega, in rad/us (constant).
    detuning : callable
        Detuning profile Delta(t), mapping time(s) in us to rad/us, e.g.
        ``lambda t: cubic_detuning(t, delta_max, t_max)``.
    phi : float, default 0.0
        Laser phase, in rad (constant).
    v : float, default 0.0
        Van der Waals shift V of |rr>, in rad/us (constant).

    Returns
    -------
    callable
        ``h(t)`` returning H(t) / hbar in rad/us, shape ``(4, 4)`` for scalar
        ``t`` or ``t.shape + (4, 4)`` for an array of times.
    """
    h_static = two_atom_hamiltonian(omega, 0.0, phi, v)
    dh_ddelta = two_atom_hamiltonian(0.0, 1.0, 0.0, 0.0)

    def hamiltonian(t: ArrayLike) -> ComplexArray:
        delta = np.asarray(detuning(t), dtype=np.float64)
        h: ComplexArray = h_static + delta[..., None, None] * dh_ddelta
        return h

    return hamiltonian


def landau_zener_bell_population(
    omega: float, delta_max: float, t_max: ArrayLike
) -> RealArray:
    """Return the Landau-Zener estimate of P(|Psi+>) after a linear sweep.

    In the blockade limit, |00> with
    energy Delta couples only to |Psi+> with energy 0, through the matrix
    element g = Omega / sqrt(2) (see
    :func:`rydberg.analytic.blockade_amplitudes`). Their diabatic energies
    cross at a constant rate |d Delta / dt| = 2 Delta_max / t_max for
    :func:`linear_detuning`. The Landau-Zener formula gives the probability of
    staying in the diabatic state |00> as exp(-2 pi g^2 / |d Delta / dt|), so

        P(|Psi+>) = 1 - exp(-pi Omega^2 t_max / (2 Delta_max)).

    This assumes an infinite sweep range. For a finite range +-Delta_max the
    exact result oscillates about this value by an amount that shrinks as
    Omega / Delta_max does.

    It also assumes |rr> stays far off resonance throughout the sweep, i.e.
    |V - 2 Delta(t)| >> Omega for all t. |rr> becomes resonant with |00> (the
    facilitation condition 2 Delta = V) at Delta = V / 2, so this requires
    V - 2 Delta_max >> Omega. For V = 2 Delta_max (e.g. V/2pi = 100 MHz with
    Delta_max/2pi = 50 MHz) that resonance is reached only at t = t_max; in
    that case P(|rr>) at t_max is ~2e-4 and the estimate still holds well.

    Parameters
    ----------
    omega : float
        Rabi frequency Omega, in rad/us.
    delta_max : float
        Sweep amplitude Delta_max, in rad/us. Must be positive.
    t_max : float or array_like
        Sweep duration(s), in us.

    Returns
    -------
    numpy.ndarray
        Estimated final population of |Psi+> (dimensionless), same shape as
        ``t_max``.

    Raises
    ------
    ValueError
        If ``delta_max`` is not positive.
    """
    if delta_max <= 0:
        raise ValueError(f"delta_max must be positive, got {delta_max}.")
    duration = np.asarray(t_max, dtype=np.float64)
    population: RealArray = 1.0 - np.exp(-np.pi * omega**2 * duration / (2 * delta_max))
    return population


def instantaneous_spectrum(
    hamiltonian: TimeDependentHamiltonian,
    times: ArrayLike,
    subspace: ArrayLike | None = None,
) -> tuple[RealArray, ComplexArray]:
    """Return the eigenenergies and eigenvectors of H(t) at each time.

    Parameters
    ----------
    hamiltonian : callable
        Vectorised H(t), mapping an array of times in us to matrices
        H / hbar in rad/us of shape ``(n_times, n, n)``, as returned by
        :func:`detuning_sweep_hamiltonian`.
    times : array_like, shape (n_times,)
        Times, in us.
    subspace : array_like, shape (n, k), optional
        Orthonormal columns spanning an invariant subspace of every H(t), e.g.
        the exchange-symmetric states {|00>, |Psi+>, |rr>}. H is projected onto
        it first, which removes decoupled states such as the dark state |Psi->
        that would otherwise appear as spurious level crossings.

    Returns
    -------
    energies : numpy.ndarray, shape (n_times, k)
        Instantaneous (adiabatic) energies in ascending order, in rad/us.
    vectors : numpy.ndarray, shape (n_times, n, k), complex
        Corresponding eigenvectors as columns, in the full basis.
    """
    t = np.asarray(times, dtype=np.float64)
    h = np.asarray(hamiltonian(t), dtype=np.complex128)
    if subspace is None:
        energies, full_vectors = np.linalg.eigh(h)
        return energies, full_vectors.astype(np.complex128)
    b = np.asarray(subspace, dtype=np.complex128)
    h_sub = b.conj().T @ h @ b
    energies, v_sub = np.linalg.eigh(h_sub)
    vectors: ComplexArray = b @ v_sub
    return energies, vectors


def adiabaticity_parameter(
    hamiltonian: TimeDependentHamiltonian,
    times: ArrayLike,
    level: int = 0,
    subspace: ArrayLike | None = None,
    fd_step: float = 1e-6,
) -> RealArray:
    """Return the adiabaticity parameter of eigenstate ``level`` along a sweep.

    A(t) = max over n != level of |<n(t)| dH/dt |level(t)>| / (E_n - E_level)^2.

    The adiabatic theorem requires A << 1 for a state to follow the
    instantaneous eigenstate; with hbar = 1 the ratio is dimensionless.
    dH/dt is found by a central finite difference of step ``fd_step`` (exact
    for a linear sweep, error O(fd_step^2) otherwise).

    Parameters
    ----------
    hamiltonian : callable
        Vectorised H(t), in rad/us, as for :func:`instantaneous_spectrum`.
    times : array_like, shape (n_times,)
        Times, in us.
    level : int, default 0
        Index of the followed eigenstate in ascending energy order.
    subspace : array_like, shape (n, k), optional
        Invariant subspace to project onto, as for
        :func:`instantaneous_spectrum`.
    fd_step : float, default 1e-6
        Finite-difference step for dH/dt, in us.

    Returns
    -------
    numpy.ndarray, shape (n_times,)
        A(t), dimensionless.
    """
    t = np.asarray(times, dtype=np.float64)
    energies, vectors = instantaneous_spectrum(hamiltonian, t, subspace)
    dh_dt = (
        np.asarray(hamiltonian(t + fd_step)) - np.asarray(hamiltonian(t - fd_step))
    ) / (2 * fd_step)
    # Matrix elements <n| dH/dt |level> for all n, shape (n_times, k).
    coupling = np.einsum("tin,tij,tj->tn", vectors.conj(), dh_dt, vectors[:, :, level])
    gap = energies - energies[:, [level]]
    others = np.arange(energies.shape[1]) != level
    ratio: RealArray = np.max(np.abs(coupling[:, others]) / gap[:, others] ** 2, axis=1)
    return ratio
