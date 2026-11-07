"""Propagation for time-independent Hamiltonians via eigendecomposition.

Evolves states as |Psi(t)> = R^dagger D(t) R |Psi(0)>, where R diagonalises H
and D(t) holds the phase factors exp(-i E_n t).

Notation follows handbook Eq. (5.6): R is the unitary matrix whose *rows* are
the eigenvectors of H (as bras), so that R H R^dagger = diag(E_1, ..., E_n).
With hbar = 1, energies are angular frequencies in rad/us and times are in us,
so every product E t is a phase in rad.

The functions work for any n x n Hermitian matrix, not only the two-atom 4 x 4
case.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

HERMITIAN_RTOL = 1e-12
"""Tolerance on ||H - H^dagger|| relative to ||H|| for accepting H as Hermitian."""


def _as_hamiltonian(h: ArrayLike) -> ComplexArray:
    """Convert ``h`` to a complex square matrix and check it is Hermitian."""
    matrix = np.asarray(h, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Hamiltonian must be a square matrix, got {matrix.shape}.")
    scale = max(float(np.linalg.norm(matrix)), 1.0)
    if np.linalg.norm(matrix - matrix.conj().T) > HERMITIAN_RTOL * scale:
        raise ValueError("Hamiltonian is not Hermitian.")
    return matrix


def _phases(energies: RealArray, times: ArrayLike) -> ComplexArray:
    """Return exp(-i E_j t) with shape ``times.shape + (n,)``.

    Parameters
    ----------
    energies : numpy.ndarray, shape (n,)
        Eigenenergies E_j, in rad/us.
    times : float or array_like
        Time(s) t, in us.

    Returns
    -------
    numpy.ndarray, complex
        The diagonal of D(t) for each time. Dimensionless.
    """
    t = np.asarray(times, dtype=np.float64)
    phases: ComplexArray = np.exp(-1j * np.multiply.outer(t, energies))
    return phases


def diagonalise(h: ArrayLike) -> tuple[RealArray, ComplexArray]:
    """Return the eigenenergies of ``h`` and the rotation matrix R.

    Parameters
    ----------
    h : array_like, shape (n, n)
        Hermitian Hamiltonian H / hbar, in rad/us.

    Returns
    -------
    energies : numpy.ndarray, shape (n,), float
        Eigenenergies E_j / hbar in ascending order, in rad/us.
    r : numpy.ndarray, shape (n, n), complex
        Unitary rotation matrix R of handbook Eq. (5.6), whose j-th row is the
        bra <E_j|, so that ``r @ h @ r.conj().T`` equals ``np.diag(energies)``.

    Raises
    ------
    ValueError
        If ``h`` is not square or not Hermitian.

    Notes
    -----
    ``np.linalg.eigh`` returns the eigenvectors as the *columns* of a unitary
    matrix U, with H = U diag(E) U^dagger. The handbook's R is therefore
    U^dagger. ``eigh`` is used rather than ``eig`` because it exploits
    Hermiticity: the eigenvalues are guaranteed real and the eigenvectors
    orthonormal even within a degenerate eigenspace (e.g. V = 0, Delta = 0,
    where E = 0 is doubly degenerate). ``eig`` makes no such guarantee, and a
    non-orthogonal eigenbasis would make R^dagger differ from R^-1.
    """
    matrix = _as_hamiltonian(h)
    energies, u = np.linalg.eigh(matrix)
    r: ComplexArray = u.conj().T
    return energies, r


def propagator(h: ArrayLike, times: ArrayLike) -> ComplexArray:
    """Return the time-evolution operator U(t) = R^dagger D(t) R.

    Parameters
    ----------
    h : array_like, shape (n, n)
        Time-independent Hermitian Hamiltonian H / hbar, in rad/us.
    times : float or array_like, shape (n_times,)
        Evolution time(s) t, in us.

    Returns
    -------
    numpy.ndarray, complex
        U(t) = exp(-i H t), shape ``(n, n)`` for scalar ``times`` or
        ``(n_times, n, n)`` for an array of times. Dimensionless and unitary.

    Raises
    ------
    ValueError
        If ``h`` is not square or not Hermitian.

    Notes
    -----
    H is diagonalised once; D(t) is built for every time at once by
    broadcasting. Element-wise, U_ik(t) = sum_j conj(R_ji) exp(-i E_j t) R_jk.
    """
    energies, r = diagonalise(h)
    phases = _phases(energies, times)
    u: ComplexArray = np.einsum("ji,...j,jk->...ik", r.conj(), phases, r)
    return u


def evolve(h: ArrayLike, psi0: ArrayLike, times: ArrayLike) -> ComplexArray:
    """Evolve ``psi0`` under a time-independent Hamiltonian (Eq. 5.6).

    Computes |Psi(t)> = R^dagger D(t) R |Psi(0)> at each requested time.

    Parameters
    ----------
    h : array_like, shape (n, n)
        Time-independent Hermitian Hamiltonian H / hbar, in rad/us.
    psi0 : array_like, shape (n,)
        Initial state |Psi(0)> in the computational basis. Dimensionless and
        assumed normalised.
    times : float or array_like, shape (n_times,)
        Evolution time(s) t, in us.

    Returns
    -------
    numpy.ndarray, complex
        States |Psi(t)>, shape ``(n,)`` for scalar ``times`` or
        ``(n_times, n)`` for an array of times, matching the trajectory layout
        used in :mod:`rydberg.states`.

    Raises
    ------
    ValueError
        If ``h`` is not square or not Hermitian, or if the shapes of ``h`` and
        ``psi0`` are incompatible.

    Notes
    -----
    Three steps: rotate into the eigenbasis (c = R |Psi(0)>), multiply each
    coefficient c_j by exp(-i E_j t), then rotate back with R^dagger. Acting on
    the vector costs O(n^2) per time, against O(n^3) for building the full
    propagator at each time.
    """
    energies, r = diagonalise(h)
    state = np.asarray(psi0, dtype=np.complex128)
    if state.shape != (r.shape[0],):
        raise ValueError(
            f"Initial state of shape {state.shape} is incompatible with a "
            f"{r.shape[0]} x {r.shape[0]} Hamiltonian."
        )
    coefficients = r @ state  # c_j = <E_j|Psi(0)>
    evolved = _phases(energies, times) * coefficients  # c_j exp(-i E_j t)
    # Rotate back: Psi_i(t) = sum_j conj(R_ji) c_j(t), i.e. R^dagger c(t).
    psi_t: ComplexArray = evolved @ r.conj()
    return psi_t
