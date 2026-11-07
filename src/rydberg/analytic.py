"""Closed-form reference solutions used for validation.

Analytic results in limiting cases (V = 0 and |V| >> |Omega|) against which the
numerical propagators are compared. Every formula here is derived by hand from
the Hamiltonians of handbook Eqs. (5.3) and (5.4); nothing in this module calls
a numerical eigensolver, so a comparison with :mod:`rydberg.evolution` is an
independent check.

All functions return complex amplitudes in the computational basis, in the same
layout as :func:`rydberg.evolution.eigen.evolve`: shape ``(dim,)`` for a scalar
time and ``(n_times, dim)`` for an array of times. Populations, Bell-state
overlaps and fidelities then follow from :mod:`rydberg.states`.

Units: hbar = 1, times in us, Omega, Delta and V in rad/us, phi in rad.

The amplitudes (in particular the signs attached to Delta and phi) follow the
handbook Hamiltonian as written. Agreement with the numerics therefore shows
consistency with Eqs. (5.3)-(5.4), not that their sign convention is physical.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rydberg.states import ComplexArray


def _sin_over(w: float, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return sin(w t / 2) / w, with the w -> 0 limit t / 2 handled exactly.

    Parameters
    ----------
    w : float
        Angular frequency, in rad/us.
    t : numpy.ndarray
        Times, in us.

    Returns
    -------
    numpy.ndarray
        sin(w t / 2) / w, in us. Uses ``np.sinc(x) = sin(pi x) / (pi x)``.
    """
    result: NDArray[np.float64] = 0.5 * t * np.sinc(w * t / (2 * np.pi))
    return result


def single_atom_amplitudes(
    times: ArrayLike, omega: float, delta: float = 0.0, phi: float = 0.0
) -> ComplexArray:
    """Return the amplitudes (a_0, a_r) of one driven atom starting in |0>.

    For H = 1/2 [[Delta, Omega e^{-i phi}], [Omega e^{i phi}, -Delta]] write
    H = (Omega'/2) n.sigma with Omega' = sqrt(Omega^2 + Delta^2). Since
    (n.sigma)^2 = 1, exp(-i H t) = cos(Omega' t/2) - i sin(Omega' t/2) n.sigma,
    and acting on |0> gives

        a_0(t) = cos(Omega' t/2) - i (Delta/Omega') sin(Omega' t/2),
        a_r(t) = -i (Omega/Omega') e^{i phi} sin(Omega' t/2).

    This is exact for any Omega, Delta and phi.

    Parameters
    ----------
    times : float or array_like, shape (n_times,)
        Time(s) t, in us.
    omega : float
        Rabi frequency Omega, in rad/us.
    delta : float, default 0.0
        Detuning Delta, in rad/us.
    phi : float, default 0.0
        Laser phase phi_L, in rad.

    Returns
    -------
    numpy.ndarray, complex
        Amplitudes in the basis {|0>, |r>}, shape ``(2,)`` for scalar ``times``
        or ``(n_times, 2)`` otherwise. Dimensionless.
    """
    t = np.asarray(times, dtype=np.float64)
    w = float(np.hypot(omega, delta))  # generalised Rabi frequency Omega'
    s = _sin_over(w, t)  # sin(Omega' t/2) / Omega', finite as Omega' -> 0
    a_0 = np.cos(w * t / 2) - 1j * delta * s
    a_r = -1j * omega * np.exp(1j * phi) * s
    amplitudes: ComplexArray = np.stack([a_0, a_r], axis=-1).astype(np.complex128)
    return amplitudes


def rabi_population(times: ArrayLike, omega: float, delta: float = 0.0) -> NDArray:
    """Return the Rydberg population of one atom starting in |0>.

    P_r(t) = (Omega^2 / Omega'^2) sin^2(Omega' t / 2), with
    Omega' = sqrt(Omega^2 + Delta^2). Exact, and independent of phi.

    Parameters
    ----------
    times : float or array_like, shape (n_times,)
        Time(s) t, in us.
    omega : float
        Rabi frequency Omega, in rad/us.
    delta : float, default 0.0
        Detuning Delta, in rad/us.

    Returns
    -------
    numpy.ndarray, float
        P_r at each time, same shape as ``times``. Dimensionless, in [0, 1].
    """
    t = np.asarray(times, dtype=np.float64)
    w = float(np.hypot(omega, delta))
    population: NDArray[np.float64] = (omega * _sin_over(w, t)) ** 2
    return population


def no_interaction_amplitudes(
    times: ArrayLike, omega: float, delta: float = 0.0, phi: float = 0.0
) -> ComplexArray:
    """Return two-atom amplitudes for V = 0, starting in |00>.

    With V = 0, H = H_A (x) 1 + 1 (x) H_B and the two terms commute, so the
    evolution factorises and the state stays a product:

        a_00 = a_0^2,  a_0r = a_r0 = a_0 a_r,  a_rr = a_r^2,

    with (a_0, a_r) from :func:`single_atom_amplitudes`. At Delta = 0 this gives
    P_00 = cos^4(Omega t/2) and P_Psi+ = sin^2(Omega t) / 2. Exact for any
    Omega, Delta and phi.

    Parameters
    ----------
    times : float or array_like, shape (n_times,)
        Time(s) t, in us.
    omega : float
        Rabi frequency Omega, in rad/us.
    delta : float, default 0.0
        Detuning Delta, in rad/us.
    phi : float, default 0.0
        Laser phase phi_L, in rad.

    Returns
    -------
    numpy.ndarray, complex
        Amplitudes in the basis {|00>, |0r>, |r0>, |rr>}, shape ``(4,)`` for
        scalar ``times`` or ``(n_times, 4)`` otherwise. Dimensionless.
    """
    a = single_atom_amplitudes(times, omega, delta, phi)
    # Outer product a_i a_j per time, flattened row-major: index 2 i + j, which
    # matches the Kronecker ordering |00>, |0r>, |r0>, |rr>.
    pair = a[..., :, None] * a[..., None, :]
    amplitudes: ComplexArray = pair.reshape(*a.shape[:-1], 4)
    return amplitudes


def blockade_amplitudes(
    times: ArrayLike, omega: float, delta: float = 0.0, phi: float = 0.0
) -> ComplexArray:
    """Return two-atom amplitudes in the blockade limit, starting in |00>.

    For |V| >> |Omega|, |Delta| the state |rr> is far off resonance and is
    dropped. |00> (energy Delta) then couples only to |Psi+> (energy 0) with
    matrix element <00|H|Psi+> = Omega e^{-i phi} / sqrt(2): a two-level system
    with collective Rabi frequency sqrt(2) Omega. With W = sqrt(2 Omega^2 +
    Delta^2),

        a_00   = e^{-i Delta t/2} [cos(W t/2) - i (Delta/W) sin(W t/2)],
        a_Psi+ = -i e^{-i Delta t/2} (sqrt(2) Omega / W) e^{i phi} sin(W t/2),

    and a_0r = a_r0 = a_Psi+ / sqrt(2), a_rr = 0. At Delta = 0 this gives
    P_00 = cos^2(sqrt(2) Omega t/2) and P_Psi+ = sin^2(sqrt(2) Omega t/2).

    This is an approximation, not an exact solution, so residuals against the
    full evolution measure the validity of the blockade picture, not numerical
    error. The dominant neglected effect is the second-order (light) shift of
    |Psi+> by about -Omega^2 / (2 (V - Delta)) from its coupling to |rr>. It
    accumulates a phase of order Omega/V over a Rabi period, so amplitude
    residuals scale as Omega/V. At Delta = 0 it acts on populations only at
    second order, giving population residuals of order (Omega/V)^2; at
    Delta != 0 population residuals are again of order Omega/V.

    Parameters
    ----------
    times : float or array_like, shape (n_times,)
        Time(s) t, in us.
    omega : float
        Rabi frequency Omega, in rad/us.
    delta : float, default 0.0
        Detuning Delta, in rad/us.
    phi : float, default 0.0
        Laser phase phi_L, in rad.

    Returns
    -------
    numpy.ndarray, complex
        Amplitudes in the basis {|00>, |0r>, |r0>, |rr>}, shape ``(4,)`` for
        scalar ``times`` or ``(n_times, 4)`` otherwise. Dimensionless.
    """
    t = np.asarray(times, dtype=np.float64)
    w = float(np.hypot(np.sqrt(2) * omega, delta))
    s = _sin_over(w, t)  # sin(W t/2) / W
    global_phase = np.exp(-0.5j * delta * t)
    a_00 = global_phase * (np.cos(w * t / 2) - 1j * delta * s)
    a_psi_plus = -1j * global_phase * np.sqrt(2) * omega * np.exp(1j * phi) * s
    a_single = a_psi_plus / np.sqrt(2)  # each of a_0r, a_r0
    amplitudes: ComplexArray = np.stack(
        [a_00, a_single, a_single, np.zeros_like(a_00)], axis=-1
    ).astype(np.complex128)
    return amplitudes
