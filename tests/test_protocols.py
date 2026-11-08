"""Tests for the detuning-sweep profiles and the time-dependent Hamiltonian."""

import numpy as np
import pytest

from rydberg.hamiltonians import two_atom_hamiltonian
from rydberg.protocols import (
    SWEEP_PROFILES,
    adiabaticity_parameter,
    cubic_detuning,
    detuning_sweep_hamiltonian,
    instantaneous_spectrum,
    landau_zener_bell_population,
    linear_detuning,
)
from rydberg.states import KET_00, KET_RR, PSI_PLUS
from rydberg.units import mhz_to_angular

DELTA_MAX = mhz_to_angular(50.0)
"""Sweep amplitude of handbook Fig. 5.2, in rad/us."""

T_MAX = 1.0
"""Sweep duration of handbook Fig. 5.2, in us."""

ATOL = 1e-10
"""Absolute tolerance on detunings (rad/us) for exact identities."""

PROFILES = list(SWEEP_PROFILES.values())


@pytest.mark.parametrize("profile", PROFILES)
def test_endpoints_are_minus_and_plus_delta_max(profile) -> None:
    assert profile(0.0, DELTA_MAX, T_MAX) == pytest.approx(-DELTA_MAX, abs=ATOL)
    assert profile(T_MAX, DELTA_MAX, T_MAX) == pytest.approx(DELTA_MAX, abs=ATOL)


@pytest.mark.parametrize("profile", PROFILES)
def test_resonance_at_half_time(profile) -> None:
    assert profile(T_MAX / 2, DELTA_MAX, T_MAX) == pytest.approx(0.0, abs=ATOL)


@pytest.mark.parametrize("profile", PROFILES)
def test_odd_about_half_time(profile) -> None:
    tau = np.linspace(0.0, T_MAX / 2, 51)
    left = profile(T_MAX / 2 - tau, DELTA_MAX, T_MAX)
    right = profile(T_MAX / 2 + tau, DELTA_MAX, T_MAX)
    np.testing.assert_allclose(left, -right, atol=ATOL)


@pytest.mark.parametrize("profile", PROFILES)
def test_monotonic_increasing(profile) -> None:
    delta = profile(np.linspace(0.0, T_MAX, 201), DELTA_MAX, T_MAX)
    assert np.all(np.diff(delta) >= 0)


@pytest.mark.parametrize("profile", PROFILES)
def test_accepts_scalars_and_arrays(profile) -> None:
    times = np.linspace(0.0, T_MAX, 7)
    assert np.shape(profile(0.3, DELTA_MAX, T_MAX)) == ()
    assert profile(times, DELTA_MAX, T_MAX).shape == times.shape


def test_cubic_has_zero_slope_at_resonance() -> None:
    """Central finite difference: dDelta/dt ~ 24 Delta_max h^2 / t_max^3 -> 0."""
    h = 1e-4
    slope = (
        cubic_detuning(T_MAX / 2 + h, DELTA_MAX, T_MAX)
        - cubic_detuning(T_MAX / 2 - h, DELTA_MAX, T_MAX)
    ) / (2 * h)
    linear_slope = 2 * DELTA_MAX / T_MAX
    assert abs(slope) < 1e-6 * linear_slope


def test_linear_has_constant_slope() -> None:
    times = np.linspace(0.0, T_MAX, 11)
    slopes = np.diff(linear_detuning(times, DELTA_MAX, T_MAX)) / np.diff(times)
    np.testing.assert_allclose(slopes, 2 * DELTA_MAX / T_MAX, rtol=1e-12)


def test_cubic_formula_value() -> None:
    """Spot value of 8 Delta_max [(t - t_max/2) / t_max]^3 at t = t_max/4."""
    assert cubic_detuning(0.25, DELTA_MAX, T_MAX) == pytest.approx(-DELTA_MAX / 8)


@pytest.mark.parametrize("profile", PROFILES)
def test_rejects_non_positive_duration(profile) -> None:
    with pytest.raises(ValueError, match="t_max"):
        profile(0.1, DELTA_MAX, 0.0)


def test_sweep_hamiltonian_matches_two_atom_hamiltonian() -> None:
    omega, phi, v = mhz_to_angular(1.0), 0.7, mhz_to_angular(20.0)
    h_of_t = detuning_sweep_hamiltonian(
        omega, lambda t: cubic_detuning(t, DELTA_MAX, T_MAX), phi, v
    )
    times = np.linspace(0.0, T_MAX, 13)
    stacked = h_of_t(times)
    assert stacked.shape == (times.size, 4, 4)
    for t, h in zip(times, stacked, strict=True):
        expected = two_atom_hamiltonian(
            omega, float(cubic_detuning(t, DELTA_MAX, T_MAX)), phi, v
        )
        np.testing.assert_allclose(h, expected, atol=ATOL)
        np.testing.assert_allclose(h_of_t(float(t)), expected, atol=ATOL)


def test_landau_zener_limits() -> None:
    omega = mhz_to_angular(1.0)
    assert landau_zener_bell_population(omega, DELTA_MAX, 0.0) == pytest.approx(0.0)
    assert landau_zener_bell_population(omega, DELTA_MAX, 1e4) == pytest.approx(1.0)
    # 1 - exp(-pi Omega^2 t_max / (2 Delta_max)) = 1 - exp(-pi^2/50 * t_max).
    assert landau_zener_bell_population(omega, DELTA_MAX, 1.0) == pytest.approx(
        1 - np.exp(-(np.pi**2) / 50)
    )


# --- adiabatic analysis -------------------------------------------------------

BRIGHT = np.stack([KET_00, PSI_PLUS, KET_RR], axis=-1)
"""Exchange-symmetric subspace {|00>, |Psi+>, |rr>} as columns."""


def test_spectrum_far_from_resonance_is_diabatic() -> None:
    """At Delta = -Delta_max the levels are close to |00>, |Psi+>, |rr>."""
    omega, v = mhz_to_angular(1.0), mhz_to_angular(100.0)
    h = detuning_sweep_hamiltonian(omega, lambda t: cubic_detuning(t, DELTA_MAX, T_MAX), 0.0, v)
    energies, vectors = instantaneous_spectrum(h, np.array([0.0]), BRIGHT)
    # Diabatic energies Delta, 0, V - Delta, shifted by at most ~Omega^2/Delta_max.
    np.testing.assert_allclose(
        energies[0], [-DELTA_MAX, 0.0, v + DELTA_MAX], atol=omega**2 / DELTA_MAX
    )
    assert abs(vectors[0, 0, 0]) ** 2 > 0.99  # ground state is mostly |00>


def test_spectrum_full_space_matches_eigvalsh() -> None:
    omega, v = mhz_to_angular(1.0), mhz_to_angular(20.0)
    h = detuning_sweep_hamiltonian(omega, lambda t: linear_detuning(t, DELTA_MAX, T_MAX), 0.3, v)
    times = np.linspace(0.0, T_MAX, 5)
    energies, _ = instantaneous_spectrum(h, times)
    np.testing.assert_allclose(energies, np.linalg.eigvalsh(h(times)), atol=ATOL)


def test_adiabaticity_parameter_at_resonance() -> None:
    """Blockaded two-level crossing: at Delta = 0, A = |dDelta/dt| / (4 Omega^2).

    The eigenstates are (|00> +- |Psi+>)/sqrt(2) with gap sqrt(2) Omega, and
    dH/dt = Delta' |00><00| on this pair, so |<e|dH/dt|g>| = Delta'/2. The
    cubic sweep has Delta' = 0 there, so A = 0.
    """
    omega, v = mhz_to_angular(1.0), mhz_to_angular(1.0e5)
    mid = np.array([T_MAX / 2])
    linear = detuning_sweep_hamiltonian(omega, lambda t: linear_detuning(t, DELTA_MAX, T_MAX), 0.0, v)
    cubic = detuning_sweep_hamiltonian(omega, lambda t: cubic_detuning(t, DELTA_MAX, T_MAX), 0.0, v)
    expected = (2 * DELTA_MAX / T_MAX) / (4 * omega**2)
    assert adiabaticity_parameter(linear, mid, subspace=BRIGHT)[0] == pytest.approx(
        expected, rel=1e-3
    )
    assert adiabaticity_parameter(cubic, mid, subspace=BRIGHT)[0] < 1e-8
