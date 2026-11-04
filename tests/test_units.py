"""Tests for the unit conventions and conversion helpers."""

import numpy as np
import pytest
from scipy import constants

from rydberg import units


def test_hbar_is_one() -> None:
    assert units.HBAR == 1.0


def test_one_mhz_is_two_pi_rad_per_us() -> None:
    assert units.mhz_to_angular(1.0) == pytest.approx(2 * np.pi)


def test_one_ghz_is_two_thousand_pi_rad_per_us() -> None:
    assert units.ghz_to_angular(1.0) == pytest.approx(2000 * np.pi)


@pytest.mark.parametrize(
    ("forward", "backward"),
    [
        (units.mhz_to_angular, units.angular_to_mhz),
        (units.ghz_to_angular, units.angular_to_ghz),
        (units.angular_to_joules, units.joules_to_angular),
    ],
)
def test_conversions_round_trip(forward, backward) -> None:
    values = np.array([0.0, 0.5, 1.0, 50.0, 100.0])
    np.testing.assert_allclose(backward(forward(values)), values)


def test_arrays_keep_their_shape() -> None:
    f = np.linspace(0.0, 100.0, 11)
    omega = units.mhz_to_angular(f)
    assert isinstance(omega, np.ndarray)
    assert omega.shape == f.shape


def test_energy_of_one_mhz_is_planck_times_one_mhz() -> None:
    # E = hbar * omega = h * f, with f = 1 MHz.
    omega = units.mhz_to_angular(1.0)
    assert units.angular_to_joules(omega) == pytest.approx(constants.h * 1.0e6)
