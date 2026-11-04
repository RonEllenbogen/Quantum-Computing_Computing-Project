"""Tests for the Hamiltonian constructors: hermiticity, matrix elements, V = 0 limit."""

import numpy as np
import pytest

from rydberg.hamiltonians import single_atom_hamiltonian, two_atom_hamiltonian
from rydberg.states import KET_00, KET_RR, PSI_MINUS, PSI_PLUS

Params = tuple[float, float, float, float]


def eq_5_4(omega: float, delta: float, phi: float, v: float) -> np.ndarray:
    """Handbook Eq. (5.4) written out element by element.

    The module builds H from Kronecker products, so this independent
    construction cross-checks it.
    """
    a = omega * np.exp(-1j * phi)
    b = omega * np.exp(1j * phi)
    return 0.5 * np.array(
        [
            [2 * delta, a, a, 0],
            [b, 0, 0, a],
            [b, 0, 0, a],
            [0, b, b, -2 * delta + 2 * v],
        ],
        dtype=np.complex128,
    )


# --- Single atom ------------------------------------------------------------


def test_single_atom_shape_and_dtype() -> None:
    h = single_atom_hamiltonian(1.0, 0.5, 0.3)
    assert h.shape == (2, 2)
    assert h.dtype == np.complex128


def test_single_atom_matrix_elements_match_eq_5_3() -> None:
    omega, delta, phi = 2.0, 0.6, 0.4
    h = single_atom_hamiltonian(omega, delta, phi)
    assert h[0, 0] == pytest.approx(delta / 2)
    assert h[1, 1] == pytest.approx(-delta / 2)
    assert h[0, 1] == pytest.approx(omega * np.exp(-1j * phi) / 2)
    assert h[1, 0] == pytest.approx(omega * np.exp(1j * phi) / 2)


def test_single_atom_is_hermitian(params: Params) -> None:
    omega, delta, phi, _ = params
    h = single_atom_hamiltonian(omega, delta, phi)
    np.testing.assert_allclose(h, h.conj().T)


def test_single_atom_eigenvalues_are_half_generalised_rabi(params: Params) -> None:
    omega, delta, phi, _ = params
    w = np.hypot(omega, delta)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(single_atom_hamiltonian(omega, delta, phi)),
        [-w / 2, w / 2],
    )


# --- Two atoms --------------------------------------------------------------


def test_two_atom_shape_and_dtype() -> None:
    h = two_atom_hamiltonian(1.0, 0.5, 0.3, 2.0)
    assert h.shape == (4, 4)
    assert h.dtype == np.complex128


def test_two_atom_matches_eq_5_4_element_by_element(params: Params) -> None:
    np.testing.assert_allclose(two_atom_hamiltonian(*params), eq_5_4(*params))


def test_two_atom_is_hermitian(params: Params) -> None:
    h = two_atom_hamiltonian(*params)
    np.testing.assert_allclose(h, h.conj().T)


def test_two_atom_without_interaction_is_kronecker_sum(params: Params) -> None:
    omega, delta, phi, _ = params
    h1 = single_atom_hamiltonian(omega, delta, phi)
    identity = np.eye(2)
    np.testing.assert_allclose(
        two_atom_hamiltonian(omega, delta, phi, v=0.0),
        np.kron(h1, identity) + np.kron(identity, h1),
    )


def test_interaction_only_shifts_rr(params: Params) -> None:
    omega, delta, phi, v = params
    difference = two_atom_hamiltonian(omega, delta, phi, v) - two_atom_hamiltonian(
        omega, delta, phi, 0.0
    )
    expected = np.zeros((4, 4))
    expected[3, 3] = v
    np.testing.assert_allclose(difference, expected, atol=1e-12)


def test_two_atom_is_symmetric_under_atom_exchange(params: Params) -> None:
    swap = np.eye(4)[[0, 2, 1, 3]]  # exchanges |0r> and |r0>
    h = two_atom_hamiltonian(*params)
    np.testing.assert_allclose(swap @ h @ swap, h)


def test_psi_minus_is_dark(params: Params) -> None:
    h = two_atom_hamiltonian(*params)
    assert np.vdot(PSI_MINUS, h @ KET_00) == pytest.approx(0.0, abs=1e-12)
    assert np.vdot(PSI_MINUS, h @ KET_RR) == pytest.approx(0.0, abs=1e-12)


def test_psi_plus_coupling_is_root_two_enhanced(params: Params) -> None:
    omega, _, phi, _ = params
    h = two_atom_hamiltonian(*params)
    # <Psi+|H|00> = (sqrt(2) Omega / 2) e^{i phi}: a Rabi frequency sqrt(2) Omega.
    expected = np.sqrt(2) * omega / 2 * np.exp(1j * phi)
    assert np.vdot(PSI_PLUS, h @ KET_00) == pytest.approx(expected)


def test_two_atom_eigenvalues_without_interaction(params: Params) -> None:
    omega, delta, phi, _ = params
    w = np.hypot(omega, delta)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(two_atom_hamiltonian(omega, delta, phi, v=0.0)),
        [-w, 0.0, 0.0, w],
        atol=1e-12,
    )


def test_trace_equals_interaction(params: Params) -> None:
    *_, v = params
    assert np.trace(two_atom_hamiltonian(*params)) == pytest.approx(v)
