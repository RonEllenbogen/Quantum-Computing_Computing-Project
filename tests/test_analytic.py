"""Tests for the analytic reference solutions against numerical evolution.

The analytic module never calls an eigensolver, so agreement with
:func:`rydberg.evolution.eigen.evolve` is an independent cross-check. It shows
consistency with handbook Eqs. (5.3)-(5.4) as written; it does not verify that
their sign convention for Delta and phi is physical.
"""

import numpy as np
import pytest

from rydberg.analytic import (
    blockade_amplitudes,
    no_interaction_amplitudes,
    rabi_population,
    single_atom_amplitudes,
)
from rydberg.evolution.eigen import evolve
from rydberg.hamiltonians import single_atom_hamiltonian, two_atom_hamiltonian
from rydberg.states import KET_00, KET_RR, PSI_PLUS, norm, overlap_mod_squared
from rydberg.units import mhz_to_angular

Params = tuple[float, float, float, float]

ATOL = 1e-10
"""Absolute tolerance for exact closed forms, set by floating-point round-off."""

OMEGA = mhz_to_angular(1.0)
"""Rabi frequency Omega / 2pi = 1 MHz, in rad/us."""

TIMES = np.linspace(0.0, 3.0, 301)
"""Evolution times in us: three Rabi periods at Omega / 2pi = 1 MHz."""

KET_0 = np.array([1.0, 0.0], dtype=np.complex128)
"""Single-atom ground state |0>."""


# --- Single atom --------------------------------------------------------------


def test_single_atom_matches_numerics(params: Params) -> None:
    omega, delta, phi, _ = params
    numerical = evolve(single_atom_hamiltonian(omega, delta, phi), KET_0, TIMES)
    analytic = single_atom_amplitudes(TIMES, omega, delta, phi)
    np.testing.assert_allclose(analytic, numerical, atol=ATOL)


def test_single_atom_shapes() -> None:
    assert single_atom_amplitudes(0.4, OMEGA).shape == (2,)
    assert single_atom_amplitudes(TIMES, OMEGA).shape == (TIMES.size, 2)


def test_single_atom_is_normalised(params: Params) -> None:
    omega, delta, phi, _ = params
    np.testing.assert_allclose(norm(single_atom_amplitudes(TIMES, omega, delta, phi)), 1.0)


def test_single_atom_without_driving_is_static() -> None:
    """Omega = Delta = 0 exercises the Omega' -> 0 limit of sin(Omega' t/2) / Omega'."""
    amplitudes = single_atom_amplitudes(TIMES, 0.0, 0.0)
    assert np.all(np.isfinite(amplitudes))
    np.testing.assert_allclose(amplitudes, np.broadcast_to(KET_0, amplitudes.shape))


def test_single_atom_detuning_only_gives_phase() -> None:
    """With Omega = 0, |0> has energy Delta/2 and only picks up exp(-i Delta t/2)."""
    delta = mhz_to_angular(3.0)
    amplitudes = single_atom_amplitudes(TIMES, 0.0, delta)
    np.testing.assert_allclose(amplitudes[:, 0], np.exp(-0.5j * delta * TIMES), atol=ATOL)
    np.testing.assert_allclose(amplitudes[:, 1], 0.0, atol=ATOL)


def test_rabi_population_is_rydberg_amplitude_squared(params: Params) -> None:
    omega, delta, phi, _ = params
    amplitudes = single_atom_amplitudes(TIMES, omega, delta, phi)
    np.testing.assert_allclose(
        rabi_population(TIMES, omega, delta), np.abs(amplitudes[:, 1]) ** 2, atol=ATOL
    )


def test_resonant_pi_pulse_fully_excites() -> None:
    assert rabi_population(np.pi / OMEGA, OMEGA) == pytest.approx(1.0)
    assert rabi_population(2 * np.pi / OMEGA, OMEGA) == pytest.approx(0.0, abs=ATOL)


def test_detuned_rabi_amplitude_is_reduced() -> None:
    """Peak population Omega^2 / (Omega^2 + Delta^2), reached at t = pi / Omega'."""
    delta = 2.0 * OMEGA
    w = np.hypot(OMEGA, delta)
    assert rabi_population(np.pi / w, OMEGA, delta) == pytest.approx(
        OMEGA**2 / (OMEGA**2 + delta**2)
    )


def test_rabi_population_is_even_in_detuning() -> None:
    delta = mhz_to_angular(2.0)
    np.testing.assert_allclose(
        rabi_population(TIMES, OMEGA, delta), rabi_population(TIMES, OMEGA, -delta)
    )


# --- Two atoms, V = 0 ---------------------------------------------------------


def test_no_interaction_matches_numerics(params: Params) -> None:
    """Exact for any Omega, Delta, phi once V is set to zero."""
    omega, delta, phi, _ = params
    numerical = evolve(two_atom_hamiltonian(omega, delta, phi, 0.0), KET_00, TIMES)
    analytic = no_interaction_amplitudes(TIMES, omega, delta, phi)
    np.testing.assert_allclose(analytic, numerical, atol=ATOL)


def test_no_interaction_shapes() -> None:
    assert no_interaction_amplitudes(0.4, OMEGA).shape == (4,)
    assert no_interaction_amplitudes(TIMES, OMEGA).shape == (TIMES.size, 4)


def test_no_interaction_is_product_state(params: Params) -> None:
    """A product state has a_00 a_rr = a_0r a_r0 (zero concurrence)."""
    omega, delta, phi, _ = params
    a = no_interaction_amplitudes(TIMES, omega, delta, phi)
    np.testing.assert_allclose(a[:, 0] * a[:, 3], a[:, 1] * a[:, 2], atol=ATOL)


def test_no_interaction_resonant_populations() -> None:
    """At Delta = 0: P_00 = cos^4(Omega t/2), P_Psi+ = sin^2(Omega t) / 2."""
    a = no_interaction_amplitudes(TIMES, OMEGA)
    np.testing.assert_allclose(
        overlap_mod_squared(a, KET_00), np.cos(OMEGA * TIMES / 2) ** 4, atol=ATOL
    )
    np.testing.assert_allclose(
        overlap_mod_squared(a, PSI_PLUS), np.sin(OMEGA * TIMES) ** 2 / 2, atol=ATOL
    )


# --- Two atoms, blockade limit ------------------------------------------------


def test_blockade_shapes() -> None:
    assert blockade_amplitudes(0.4, OMEGA).shape == (4,)
    assert blockade_amplitudes(TIMES, OMEGA).shape == (TIMES.size, 4)


def test_blockade_is_normalised_and_never_doubly_excited(params: Params) -> None:
    omega, delta, phi, _ = params
    a = blockade_amplitudes(TIMES, omega, delta, phi)
    np.testing.assert_allclose(norm(a), 1.0, atol=ATOL)
    np.testing.assert_allclose(overlap_mod_squared(a, KET_RR), 0.0, atol=ATOL)


def test_blockade_resonant_populations() -> None:
    """At Delta = 0 the collective Rabi frequency is sqrt(2) Omega."""
    a = blockade_amplitudes(TIMES, OMEGA)
    phase = np.sqrt(2) * OMEGA * TIMES / 2
    np.testing.assert_allclose(overlap_mod_squared(a, KET_00), np.cos(phase) ** 2, atol=ATOL)
    np.testing.assert_allclose(overlap_mod_squared(a, PSI_PLUS), np.sin(phase) ** 2, atol=ATOL)


# Ratios V / Omega spanning more than a decade, all deep in the blockade regime.
BLOCKADE_RATIOS = np.array([25.0, 50.0, 100.0, 200.0, 400.0])


def blockade_residuals(
    delta: float, phi: float, v_sign: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return max amplitude and population residuals for each V / Omega ratio.

    Residuals are max over TIMES of |numerical - analytic| for the amplitudes,
    and of |P_numerical - P_analytic| over the four basis populations.
    """
    analytic = blockade_amplitudes(TIMES, OMEGA, delta, phi)
    amplitude_errors, population_errors = [], []
    for ratio in BLOCKADE_RATIOS:
        h = two_atom_hamiltonian(OMEGA, delta, phi, v_sign * ratio * OMEGA)
        numerical = evolve(h, KET_00, TIMES)
        amplitude_errors.append(np.max(np.abs(numerical - analytic)))
        population_errors.append(np.max(np.abs(np.abs(numerical) ** 2 - np.abs(analytic) ** 2)))
    return np.array(amplitude_errors), np.array(population_errors)


def log_log_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return the least-squares slope of log(y) against log(x)."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


@pytest.mark.convergence
@pytest.mark.parametrize("v_sign", [1.0, -1.0], ids=["repulsive", "attractive"])
def test_blockade_resonant_residuals_scale_with_omega_over_v(v_sign: float) -> None:
    """At Delta = 0: amplitude residuals ~ Omega/V, population residuals ~ (Omega/V)^2.

    The neglected light shift of |Psi+> is a phase error of order Omega/V, which
    enters the populations only at second order when Delta = 0.
    """
    amplitude_errors, population_errors = blockade_residuals(0.0, 0.0, v_sign)
    assert log_log_slope(BLOCKADE_RATIOS, amplitude_errors) == pytest.approx(-1.0, abs=0.05)
    assert log_log_slope(BLOCKADE_RATIOS, population_errors) == pytest.approx(-2.0, abs=0.05)


@pytest.mark.convergence
@pytest.mark.parametrize("v_sign", [1.0, -1.0], ids=["repulsive", "attractive"])
def test_blockade_detuned_residuals_scale_with_omega_over_v(v_sign: float) -> None:
    """At Delta != 0 the light shift acts as an extra detuning, so both scale as Omega/V."""
    amplitude_errors, population_errors = blockade_residuals(
        mhz_to_angular(0.3), 0.7, v_sign
    )
    assert log_log_slope(BLOCKADE_RATIOS, amplitude_errors) == pytest.approx(-1.0, abs=0.1)
    assert log_log_slope(BLOCKADE_RATIOS, population_errors) == pytest.approx(-1.0, abs=0.1)


def test_blockade_is_good_at_handbook_parameters() -> None:
    """At V / 2pi = 100 MHz the blockade picture holds to better than 1e-3 in population."""
    v = mhz_to_angular(100.0)
    numerical = evolve(two_atom_hamiltonian(OMEGA, 0.0, 0.0, v), KET_00, TIMES)
    analytic = blockade_amplitudes(TIMES, OMEGA)
    np.testing.assert_allclose(np.abs(numerical) ** 2, np.abs(analytic) ** 2, atol=1e-3)


def test_blockade_fails_without_interaction() -> None:
    """Sanity check that the residual tests can fail: at V = 0 blockade is wrong."""
    numerical = evolve(two_atom_hamiltonian(OMEGA, 0.0, 0.0, 0.0), KET_00, TIMES)
    analytic = blockade_amplitudes(TIMES, OMEGA)
    assert np.max(np.abs(np.abs(numerical) ** 2 - np.abs(analytic) ** 2)) > 0.1
