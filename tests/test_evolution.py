"""Tests for the propagators: unitarity, norm conservation and cross-checks."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm

from rydberg.evolution.eigen import diagonalise, evolve, propagator
from rydberg.hamiltonians import two_atom_hamiltonian
from rydberg.states import BASIS, KET_00, PSI_MINUS, norm, overlap_mod_squared

Params = tuple[float, float, float, float]

ATOL = 1e-10
"""Absolute tolerance for exact identities, set by floating-point round-off."""

SEED = 20261108
"""Seed for the random Hermitian matrices used to test general dimensions."""

TIMES = np.linspace(0.0, 3.0, 61)
"""Evolution times in us: several Rabi periods for Omega / 2pi = 1 MHz."""


def random_hermitian(dim: int, rng: np.random.Generator) -> NDArray[np.complex128]:
    """Return a random dim x dim Hermitian matrix with entries of order 1 rad/us."""
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h: NDArray[np.complex128] = ((a + a.conj().T) / 2).astype(np.complex128)
    return h


def random_state(dim: int, rng: np.random.Generator) -> NDArray[np.complex128]:
    """Return a random normalised state of dimension ``dim``."""
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    normalised: NDArray[np.complex128] = (psi / np.linalg.norm(psi)).astype(np.complex128)
    return normalised


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a deterministically seeded random number generator."""
    return np.random.default_rng(SEED)


# --- diagonalise --------------------------------------------------------------


def test_diagonalise_rotates_h_to_diagonal(params: Params) -> None:
    h = two_atom_hamiltonian(*params)
    energies, r = diagonalise(h)
    np.testing.assert_allclose(r @ h @ r.conj().T, np.diag(energies), atol=ATOL)


def test_diagonalise_reconstructs_h(params: Params) -> None:
    h = two_atom_hamiltonian(*params)
    energies, r = diagonalise(h)
    np.testing.assert_allclose(r.conj().T @ np.diag(energies) @ r, h, atol=ATOL)


def test_rotation_matrix_is_unitary(params: Params) -> None:
    _, r = diagonalise(two_atom_hamiltonian(*params))
    np.testing.assert_allclose(r @ r.conj().T, np.eye(4), atol=ATOL)


def test_rows_of_r_are_eigenbras(params: Params) -> None:
    """Handbook Eq. (5.6) convention: row j of R is <E_j|, so H R^dagger_j = E_j."""
    h = two_atom_hamiltonian(*params)
    energies, r = diagonalise(h)
    for energy, bra in zip(energies, r, strict=True):
        ket = bra.conj()
        np.testing.assert_allclose(h @ ket, energy * ket, atol=ATOL)


def test_energies_are_real_and_ascending(params: Params) -> None:
    energies, _ = diagonalise(two_atom_hamiltonian(*params))
    assert energies.dtype == np.float64
    assert np.all(np.diff(energies) >= 0)


def test_diagonalise_degenerate_spectrum_stays_unitary() -> None:
    """At V = Delta = 0, E = 0 is doubly degenerate (|Psi-> and a bright state)."""
    h = two_atom_hamiltonian(2 * np.pi, 0.0, 0.0, 0.0)
    energies, r = diagonalise(h)
    assert np.sum(np.isclose(energies, 0.0, atol=ATOL)) == 2
    np.testing.assert_allclose(r @ r.conj().T, np.eye(4), atol=ATOL)


def test_diagonalise_rejects_non_hermitian() -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        diagonalise(np.array([[0.0, 1.0], [0.0, 0.0]]))


@pytest.mark.parametrize("shape", [(2, 3), (4,), (2, 2, 2)])
def test_diagonalise_rejects_non_square(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="square"):
        diagonalise(np.zeros(shape))


# --- propagator ---------------------------------------------------------------


def test_propagator_at_zero_time_is_identity(params: Params) -> None:
    u = propagator(two_atom_hamiltonian(*params), 0.0)
    np.testing.assert_allclose(u, np.eye(4), atol=ATOL)


def test_propagator_shapes(params: Params) -> None:
    h = two_atom_hamiltonian(*params)
    assert propagator(h, 0.5).shape == (4, 4)
    assert propagator(h, TIMES).shape == (TIMES.size, 4, 4)


def test_propagator_is_unitary_at_all_times(params: Params) -> None:
    u = propagator(two_atom_hamiltonian(*params), TIMES)
    u_dagger_u = np.einsum("tji,tjk->tik", u.conj(), u)
    np.testing.assert_allclose(u_dagger_u, np.broadcast_to(np.eye(4), u.shape), atol=ATOL)


def test_propagator_matches_matrix_exponential(params: Params) -> None:
    """Independent check against scipy's Pade-approximant expm(-i H t)."""
    h = two_atom_hamiltonian(*params)
    u = propagator(h, TIMES)
    for t, u_t in zip(TIMES, u, strict=True):
        np.testing.assert_allclose(u_t, expm(-1j * h * t), atol=1e-9)


def test_propagator_composition(params: Params) -> None:
    """U(t1 + t2) = U(t2) U(t1) for a time-independent H."""
    h = two_atom_hamiltonian(*params)
    t1, t2 = 0.37, 1.21
    np.testing.assert_allclose(
        propagator(h, t1 + t2), propagator(h, t2) @ propagator(h, t1), atol=ATOL
    )


def test_propagator_time_reversal(params: Params) -> None:
    """U(-t) = U(t)^dagger."""
    h = two_atom_hamiltonian(*params)
    t = 0.83
    np.testing.assert_allclose(propagator(h, -t), propagator(h, t).conj().T, atol=ATOL)


@pytest.mark.parametrize("dim", [2, 3, 6])
def test_propagator_general_dimension(dim: int, rng: np.random.Generator) -> None:
    """The module is not specific to the 4 x 4 two-atom case."""
    h = random_hermitian(dim, rng)
    np.testing.assert_allclose(propagator(h, 0.9), expm(-0.9j * h), atol=ATOL)


# --- evolve -------------------------------------------------------------------


def test_evolve_shapes(params: Params) -> None:
    h = two_atom_hamiltonian(*params)
    assert evolve(h, KET_00, 0.5).shape == (4,)
    assert evolve(h, KET_00, TIMES).shape == (TIMES.size, 4)


def test_evolve_at_zero_time_returns_initial_state(
    params: Params, rng: np.random.Generator
) -> None:
    psi0 = random_state(4, rng)
    np.testing.assert_allclose(evolve(two_atom_hamiltonian(*params), psi0, 0.0), psi0, atol=ATOL)


def test_evolve_matches_propagator(params: Params, rng: np.random.Generator) -> None:
    h = two_atom_hamiltonian(*params)
    psi0 = random_state(4, rng)
    expected = propagator(h, TIMES) @ psi0
    np.testing.assert_allclose(evolve(h, psi0, TIMES), expected, atol=ATOL)


def test_evolve_conserves_norm(params: Params, rng: np.random.Generator) -> None:
    psi_t = evolve(two_atom_hamiltonian(*params), random_state(4, rng), TIMES)
    np.testing.assert_allclose(norm(psi_t), 1.0, atol=ATOL)


def test_evolve_conserves_energy(params: Params, rng: np.random.Generator) -> None:
    """<H> is a constant of motion for a time-independent Hamiltonian."""
    h = two_atom_hamiltonian(*params)
    psi_t = evolve(h, random_state(4, rng), TIMES)
    energy = np.einsum("ti,ij,tj->t", psi_t.conj(), h, psi_t)
    np.testing.assert_allclose(energy.imag, 0.0, atol=ATOL)
    np.testing.assert_allclose(energy.real, energy[0].real, atol=ATOL)


def test_eigenstate_only_acquires_phase(params: Params) -> None:
    """An eigenstate |E_j> evolves to exp(-i E_j t) |E_j>."""
    h = two_atom_hamiltonian(*params)
    energies, r = diagonalise(h)
    for energy, bra in zip(energies, r, strict=True):
        ket = bra.conj()
        expected = np.exp(-1j * energy * TIMES)[:, None] * ket
        np.testing.assert_allclose(evolve(h, ket, TIMES), expected, atol=ATOL)


def test_psi_minus_is_stationary(params: Params) -> None:
    """|Psi-> is a zero-energy dark state of Eq. (5.4) for any parameters."""
    psi_t = evolve(two_atom_hamiltonian(*params), PSI_MINUS, TIMES)
    np.testing.assert_allclose(psi_t, np.broadcast_to(PSI_MINUS, psi_t.shape), atol=ATOL)


def test_exchange_symmetry_keeps_psi_minus_empty(params: Params) -> None:
    """|00> is exchange-symmetric, so it never populates the antisymmetric |Psi->."""
    psi_t = evolve(two_atom_hamiltonian(*params), KET_00, TIMES)
    np.testing.assert_allclose(overlap_mod_squared(psi_t, PSI_MINUS), 0.0, atol=ATOL)
    np.testing.assert_allclose(psi_t[:, 1], psi_t[:, 2], atol=ATOL)


def test_evolve_is_linear(params: Params) -> None:
    """Evolving a superposition equals the superposition of evolved basis states."""
    h = two_atom_hamiltonian(*params)
    coefficients = np.array([0.5, 0.5j, -0.5, 0.5])
    psi0 = sum(c * ket for c, ket in zip(coefficients, BASIS, strict=True))
    expected = sum(
        c * evolve(h, ket, TIMES) for c, ket in zip(coefficients, BASIS, strict=True)
    )
    np.testing.assert_allclose(evolve(h, psi0, TIMES), expected, atol=ATOL)


def test_evolve_rejects_incompatible_state() -> None:
    h = two_atom_hamiltonian(2 * np.pi, 0.0)
    with pytest.raises(ValueError, match="incompatible"):
        evolve(h, np.array([1.0, 0.0]), TIMES)
