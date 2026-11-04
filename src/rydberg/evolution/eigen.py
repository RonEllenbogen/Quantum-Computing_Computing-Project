"""Propagation for time-independent Hamiltonians via eigendecomposition.

Evolves states as |Psi(t)> = R^dagger D(t) R |Psi(0)>, where R diagonalises H
and D(t) holds the phase factors exp(-i E_n t).
"""
