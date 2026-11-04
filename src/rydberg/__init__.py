"""Numerical simulation of two driven, interacting Rydberg-atom qubits.

Conventions: hbar = 1, time in microseconds and frequencies as angular
frequencies in rad/us.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rydberg-dynamics")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
