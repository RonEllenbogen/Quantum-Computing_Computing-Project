"""Propagation for time-dependent Hamiltonians via Trotter-Suzuki stepping.

Advances states as |Psi(t + dt)> = exp(-i H(t) dt) |Psi(t)>, with the step size
dt controlling the accuracy of the approximation.
"""
