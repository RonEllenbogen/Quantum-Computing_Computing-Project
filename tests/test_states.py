"""Tests for basis and Bell states: normalisation, orthogonality and mod_squared."""

import numpy as np
import pytest

from rydberg import states
from rydberg.states import (
    BASIS,
    BASIS_LABELS,
    KET_0R,
    KET_00,
    KET_R0,
    KET_RR,
    PHI_MINUS,
    PHI_PLUS,
    PSI_MINUS,
    PSI_PLUS,
)

BELL_STATES = (PSI_PLUS, PSI_MINUS, PHI_PLUS, PHI_MINUS)


@pytest.fixture
def random_state() -> np.ndarray:
    rng = np.random.default_rng(seed=1234)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    return psi / np.linalg.norm(psi)


def test_basis_is_orthonormal() -> None:
    basis = np.array(BASIS)
    np.testing.assert_allclose(basis.conj() @ basis.T, np.eye(4))


def test_basis_order_matches_hamiltonian_rows() -> None:
    for index, ket in enumerate((KET_00, KET_0R, KET_R0, KET_RR)):
        np.testing.assert_array_equal(ket, np.eye(4)[index])
    assert len(BASIS_LABELS) == len(BASIS)


def test_bell_states_are_orthonormal() -> None:
    bell = np.array(BELL_STATES)
    np.testing.assert_allclose(bell.conj() @ bell.T, np.eye(4), atol=1e-15)


def test_psi_plus_is_equal_superposition_of_single_excitations() -> None:
    assert states.overlap_mod_squared(PSI_PLUS, KET_0R) == pytest.approx(0.5)
    assert states.overlap_mod_squared(PSI_PLUS, KET_R0) == pytest.approx(0.5)
    assert states.overlap_mod_squared(PSI_PLUS, KET_00) == pytest.approx(0.0)


@pytest.mark.parametrize("ket", BASIS + BELL_STATES)
def test_module_states_are_read_only(ket: np.ndarray) -> None:
    with pytest.raises(ValueError, match="read-only"):
        ket[0] = 5.0


def test_overlap_mod_squared_of_basis_states() -> None:
    assert states.overlap_mod_squared(KET_00, KET_00) == pytest.approx(1.0)
    assert states.overlap_mod_squared(KET_00, KET_0R) == pytest.approx(0.0)


def test_overlap_mod_squared_conjugates_target() -> None:
    # <phi|psi> with phi = |0r> + i|r0> and psi = phi: without the conjugate on
    # phi the overlap would be 1 + i^2 = 0 instead of 2.
    phi = (KET_0R + 1j * KET_R0) / np.sqrt(2)
    assert states.overlap_mod_squared(phi, phi) == pytest.approx(1.0)


def test_overlap_mod_squared_is_invariant_under_global_phase(
    random_state: np.ndarray,
) -> None:
    phased = np.exp(1j * 0.73) * random_state
    assert states.overlap_mod_squared(phased, PSI_PLUS) == pytest.approx(
        states.overlap_mod_squared(random_state, PSI_PLUS)
    )


def test_overlap_mod_squared_of_trajectory_has_one_value_per_time() -> None:
    trajectory = np.array([KET_00, PSI_PLUS, KET_0R])
    result = states.overlap_mod_squared(trajectory, PSI_PLUS)
    assert np.shape(result) == (3,)
    np.testing.assert_allclose(result, [0.0, 1.0, 0.5])


def test_overlap_mod_squared_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="Dimension mismatch"):
        states.overlap_mod_squared(KET_00, [1, 0])


def test_overlap_mod_squared_rejects_batched_target() -> None:
    with pytest.raises(ValueError, match="single state vector"):
        states.overlap_mod_squared(KET_00, np.array([KET_00, KET_RR]))


def test_basis_mod_squared_sum_to_one(random_state: np.ndarray) -> None:
    mod_squared = states.basis_mod_squared(random_state)
    assert mod_squared.shape == (4,)
    assert mod_squared.sum() == pytest.approx(1.0)


def test_basis_mod_squared_of_trajectory() -> None:
    trajectory = np.array([KET_00, PSI_PLUS])
    np.testing.assert_allclose(
        states.basis_mod_squared(trajectory),
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]],
    )


def test_norm_of_single_state_and_batch() -> None:
    assert states.norm(PSI_PLUS) == pytest.approx(1.0)
    np.testing.assert_allclose(states.norm([2 * KET_00, 3 * KET_RR]), [2.0, 3.0])


def test_normalise_returns_unit_norm_copy() -> None:
    psi = np.array([3.0, 4.0j, 0.0, 0.0])
    normalised = states.normalise(psi)
    assert states.norm(normalised) == pytest.approx(1.0)
    np.testing.assert_array_equal(psi, [3.0, 4.0j, 0.0, 0.0])


def test_normalise_batch() -> None:
    normalised = states.normalise([2 * KET_00, 5 * PSI_PLUS])
    np.testing.assert_allclose(states.norm(normalised), [1.0, 1.0])


def test_normalise_zero_state_raises() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        states.normalise(np.zeros(4))


def test_fidelity_normalises_inputs() -> None:
    assert states.fidelity(3 * PSI_PLUS, PSI_PLUS) == pytest.approx(1.0)
    assert states.fidelity(KET_0R, 2 * PSI_PLUS) == pytest.approx(0.5)


def test_fidelity_is_symmetric(random_state: np.ndarray) -> None:
    assert states.fidelity(random_state, PSI_MINUS) == pytest.approx(
        states.fidelity(PSI_MINUS, random_state)
    )
