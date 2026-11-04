"""Rydberg-atom Hamiltonians in the rotating-wave approximation.

Single-qubit and two-atom Hamiltonians for laser driving with Rabi frequency
Omega, detuning Delta and phase phi, including the van der Waals shift V = C6/R^6.
"""

import numpy as np
from numpy.typing import NDArray

from rydberg.states import KET_RR


def single_atom_hamiltonian(
    omega: float, delta: float, phi: float = 0.0
) -> NDArray[np.complex128]:
    """Return the single-atom Hamiltonian H_int / hbar of handbook Eq. (5.3).

    In the basis {|0>, |r>} and the rotating frame,

        H_int / hbar = 1/2 [[ Delta,             Omega e^{-i phi} ],
                            [ Omega e^{i phi},  -Delta            ]].

    Parameters
    ----------
    omega : float
        Rabi frequency Omega, in rad/us. Taken to be real, as in the handbook.
    delta : float
        Detuning Delta = omega_laser - omega_0, in rad/us.
    phi : float, default 0.0
        Laser phase phi_L, in rad.

    Returns
    -------
    numpy.ndarray, shape (2, 2), complex
        Hermitian matrix H_int / hbar, in rad/us.
    """
    h = np.zeros((2, 2), dtype=np.complex128)
    h[0, 0] = delta
    h[0, 1] = omega * np.exp(-1j * phi)
    h[1, 0] = omega * np.exp(1j * phi)
    h[1, 1] = -delta
    return h / 2


def two_atom_hamiltonian(
    omega: float, delta: float, phi: float = 0.0, v: float = 0.0
) -> NDArray[np.complex128]:
    """Return the two-atom Hamiltonian H / hbar of handbook Eq. (5.4).

    Both atoms are driven by the same laser, so

        H / hbar = H_A (x) 1 + 1 (x) H_B + V |rr><rr|,

    where H_A = H_B is :func:`single_atom_hamiltonian` and (x) is the tensor
    (Kronecker) product. In the basis {|00>, |0r>, |r0>, |rr>} this is

        H / hbar = 1/2 [[ 2 Delta,      Omega e^-ip,  Omega e^-ip,  0             ],
                        [ Omega e^ip,   0,            0,            Omega e^-ip   ],
                        [ Omega e^ip,   0,            0,            Omega e^-ip   ],
                        [ 0,            Omega e^ip,   Omega e^ip,   -2 Delta + 2V ]]

    with p = phi.

    Parameters
    ----------
    omega : float
        Rabi frequency Omega, in rad/us. Taken to be real, as in the handbook.
    delta : float
        Detuning Delta = omega_laser - omega_0, in rad/us.
    phi : float, default 0.0
        Laser phase phi_L, in rad.
    v : float, default 0.0
        Van der Waals shift V = C6 / (hbar R^6) of the state |rr>, in rad/us.

    Returns
    -------
    numpy.ndarray, shape (4, 4), complex
        Hermitian matrix H / hbar, in rad/us.
    """
    h1 = single_atom_hamiltonian(omega, delta, phi)  # acts on atom A
    h2 = single_atom_hamiltonian(omega, delta, phi)  # acts on atom B
    identity = np.eye(2, dtype=np.complex128)
    h: NDArray[np.complex128] = (
        np.kron(h1, identity)
        + np.kron(identity, h2)
        + v * np.outer(KET_RR, KET_RR.conj())
    )
    return h
