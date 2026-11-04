"""Basis states, Bell states and state-analysis utilities.

Covers the two-atom computational basis {|00>, |0r>, |r0>, |rr>}, Bell states
such as |Psi+> = (|0r> + |r0>)/sqrt(2), and populations, fidelities and norms.

States are complex NumPy arrays whose last axis indexes the basis in the order
above, so a single state has shape ``(4,)`` and a trajectory of states has shape
``(n_times, 4)``. The analysis functions accept either.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]


def _read_only(vector: ArrayLike) -> ComplexArray:
    """Return ``vector`` as a read-only complex array.

    The module-level states are shared by every importer, so they are locked to
    stop an in-place operation in one place from silently changing them
    everywhere. Use ``.copy()`` to get a writable version.
    """
    array = np.array(vector, dtype=np.complex128)
    array.setflags(write=False)
    return array


# Basis vectors, in the same order as the rows of the two-atom Hamiltonian.
KET_00 = _read_only([1, 0, 0, 0])
"""|00>: both atoms in the ground state."""

KET_0R = _read_only([0, 1, 0, 0])
"""|0r>: first atom in the ground state, second in the Rydberg state."""

KET_R0 = _read_only([0, 0, 1, 0])
"""|r0>: first atom in the Rydberg state, second in the ground state."""

KET_RR = _read_only([0, 0, 0, 1])
"""|rr>: both atoms in the Rydberg state."""

BASIS = (KET_00, KET_0R, KET_R0, KET_RR)
"""The computational basis, in Hamiltonian order."""

BASIS_LABELS = ("|00⟩", "|0r⟩", "|r0⟩", "|rr⟩")
"""Plot labels for :data:`BASIS`, in the same order."""

# Bell states.
PSI_PLUS = _read_only((KET_0R + KET_R0) / np.sqrt(2))
"""|Psi+> = (|0r> + |r0>)/sqrt(2): the state |00> couples to under blockade."""

PSI_MINUS = _read_only((KET_0R - KET_R0) / np.sqrt(2))
"""|Psi-> = (|0r> - |r0>)/sqrt(2): dark state for equal driving of both atoms."""

PHI_PLUS = _read_only((KET_00 + KET_RR) / np.sqrt(2))
"""|Phi+> = (|00> + |rr>)/sqrt(2)."""

PHI_MINUS = _read_only((KET_00 - KET_RR) / np.sqrt(2))
"""|Phi-> = (|00> - |rr>)/sqrt(2)."""


def _as_states(psi: ArrayLike) -> ComplexArray:
    """Convert ``psi`` to a complex array of shape ``(..., dim)``."""
    states = np.asarray(psi, dtype=np.complex128)
    if states.ndim == 0:
        raise ValueError("A state must have at least one dimension.")
    return states


def norm(psi: ArrayLike) -> float | NDArray[np.float64]:
    """Return the norm sqrt(<psi|psi>) of a state or of each state in a batch.

    Parameters
    ----------
    psi : array_like, shape (..., dim)
        State vector, or a batch of state vectors along the leading axes.

    Returns
    -------
    float or numpy.ndarray, shape (...)
        Norm of each state (dimensionless).
    """
    result: float | NDArray[np.float64] = np.linalg.norm(_as_states(psi), axis=-1)
    return result


def normalise(psi: ArrayLike) -> ComplexArray:
    """Return a normalised copy of a state or of each state in a batch.

    Parameters
    ----------
    psi : array_like, shape (..., dim)
        State vector, or a batch of state vectors along the leading axes.

    Returns
    -------
    numpy.ndarray, shape (..., dim)
        New array in which every state has unit norm.

    Raises
    ------
    ValueError
        If any state has zero norm.
    """
    states = _as_states(psi)
    norms = np.linalg.norm(states, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalise a state with zero norm.")
    normalised: ComplexArray = states / norms
    return normalised


def overlap_mod_squared(psi: ArrayLike, phi: ArrayLike) -> float | NDArray[np.float64]:
    """Return the squared modulus of the overlap, |<phi|psi>|^2.

    By the Born rule this is the probability of finding the state ``psi`` in
    ``phi``, often called the *population* of ``phi`` in atomic physics.

    Parameters
    ----------
    psi : array_like, shape (..., dim)
        State being measured, or a batch of states such as a trajectory of
        shape ``(n_times, dim)``.
    phi : array_like, shape (dim,)
        Target state, e.g. :data:`KET_00` or :data:`PSI_PLUS`.

    Returns
    -------
    float or numpy.ndarray, shape (...)
        Population of ``phi`` in each state (dimensionless). This is a
        probability only if both states are normalised.

    Raises
    ------
    ValueError
        If ``phi`` is not a single vector or the dimensions do not match.

    Notes
    -----
    The overlap conjugates ``phi``. Using ``np.dot`` without the conjugate
    gives wrong answers whenever ``phi`` has complex amplitudes.

    Examples
    --------
    >>> float(overlap_mod_squared(PSI_PLUS, KET_0R))
    0.4999999999999999
    """
    states = _as_states(psi)
    target = _as_states(phi)
    if target.ndim != 1:
        raise ValueError(
            f"phi must be a single state vector, got shape {target.shape}."
        )
    if states.shape[-1] != target.shape[0]:
        raise ValueError(
            f"Dimension mismatch: psi has dimension {states.shape[-1]}, "
            f"phi has dimension {target.shape[0]}."
        )
    overlap = states @ target.conj()
    result: float | NDArray[np.float64] = np.abs(overlap) ** 2
    return result


def basis_mod_squared(psi: ArrayLike) -> NDArray[np.float64]:
    """Return the squared modulus |psi_i|^2 of every basis amplitude.

    These are the Born-rule probabilities (populations) of the basis states.

    Parameters
    ----------
    psi : array_like, shape (..., dim)
        State vector, or a batch of state vectors along the leading axes.

    Returns
    -------
    numpy.ndarray, shape (..., dim)
        ``|psi_i|^2`` for each basis state ``i``, in :data:`BASIS` order. For a
        normalised state these sum to 1 along the last axis.
    """
    result: NDArray[np.float64] = np.abs(_as_states(psi)) ** 2
    return result


def fidelity(psi: ArrayLike, phi: ArrayLike) -> float | NDArray[np.float64]:
    """Return the fidelity |<phi|psi>|^2 between pure states.

    Unlike :func:`overlap_mod_squared`, both states are normalised first, so the result
    always lies in [0, 1]. This formula holds for pure states only; mixed states
    (density matrices) need the Uhlmann fidelity instead.

    Parameters
    ----------
    psi : array_like, shape (..., dim)
        State, or a batch of states along the leading axes.
    phi : array_like, shape (dim,)
        Reference state.

    Returns
    -------
    float or numpy.ndarray, shape (...)
        Fidelity of each state with ``phi`` (dimensionless).

    Raises
    ------
    ValueError
        If either state has zero norm, or the dimensions do not match.
    """
    return overlap_mod_squared(normalise(psi), normalise(phi))
