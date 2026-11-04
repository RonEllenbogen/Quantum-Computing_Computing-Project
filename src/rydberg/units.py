"""Unit conventions and conversion helpers.

Throughout the package hbar = 1, time is measured in microseconds and all
frequencies are angular frequencies in rad/us (so Omega/2pi = 1 MHz is 2*pi*1.0).
Energies are therefore also expressed in rad/us, since E = hbar * omega.

Parameters are usually quoted as ordinary frequencies (MHz or GHz), so these
helpers convert them to the internal convention and back. They accept scalars or
NumPy arrays and return the same type.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import constants

HBAR: float = 1.0
"""Reduced Planck constant in internal units (dimensionless)."""

TWO_PI: float = 2.0 * np.pi
"""Conversion factor from cycles to radians."""

HBAR_SI: float = constants.hbar
"""Reduced Planck constant in J s."""

_MHZ_PER_GHZ = 1.0e3
_PER_US_TO_PER_S = 1.0e6


def mhz_to_angular[T: (float, NDArray[np.floating])](f_mhz: T) -> T:
    """Convert a frequency in MHz to an angular frequency in rad/us.

    Parameters
    ----------
    f_mhz : float or numpy.ndarray
        Ordinary frequency f, in MHz (cycles per microsecond).

    Returns
    -------
    float or numpy.ndarray
        Angular frequency omega = 2 pi f, in rad/us.

    Examples
    --------
    >>> mhz_to_angular(1.0)  # Omega/2pi = 1 MHz
    6.283185307179586
    """
    return TWO_PI * f_mhz


def angular_to_mhz[T: (float, NDArray[np.floating])](omega: T) -> T:
    """Convert an angular frequency in rad/us to a frequency in MHz.

    Parameters
    ----------
    omega : float or numpy.ndarray
        Angular frequency, in rad/us.

    Returns
    -------
    float or numpy.ndarray
        Ordinary frequency f = omega / 2 pi, in MHz.
    """
    return omega / TWO_PI


def ghz_to_angular[T: (float, NDArray[np.floating])](f_ghz: T) -> T:
    """Convert a frequency in GHz to an angular frequency in rad/us.

    Useful for van der Waals coefficients, which are commonly quoted as
    C6/h in GHz um^6; this returns C6/hbar in rad/us um^6.

    Parameters
    ----------
    f_ghz : float or numpy.ndarray
        Ordinary frequency, in GHz.

    Returns
    -------
    float or numpy.ndarray
        Angular frequency, in rad/us.
    """
    return mhz_to_angular(_MHZ_PER_GHZ * f_ghz)


def angular_to_ghz[T: (float, NDArray[np.floating])](omega: T) -> T:
    """Convert an angular frequency in rad/us to a frequency in GHz.

    Parameters
    ----------
    omega : float or numpy.ndarray
        Angular frequency, in rad/us.

    Returns
    -------
    float or numpy.ndarray
        Ordinary frequency, in GHz.
    """
    return angular_to_mhz(omega) / _MHZ_PER_GHZ


def angular_to_joules[T: (float, NDArray[np.floating])](omega: T) -> T:
    """Convert an energy in internal units (rad/us) to joules.

    Parameters
    ----------
    omega : float or numpy.ndarray
        Energy expressed as an angular frequency, E / hbar, in rad/us.

    Returns
    -------
    float or numpy.ndarray
        Energy E = hbar omega, in J.
    """
    return HBAR_SI * _PER_US_TO_PER_S * omega


def joules_to_angular[T: (float, NDArray[np.floating])](energy_j: T) -> T:
    """Convert an energy in joules to internal units (rad/us).

    Parameters
    ----------
    energy_j : float or numpy.ndarray
        Energy, in J.

    Returns
    -------
    float or numpy.ndarray
        Energy expressed as an angular frequency, E / hbar, in rad/us.
    """
    return energy_j / (HBAR_SI * _PER_US_TO_PER_S)
