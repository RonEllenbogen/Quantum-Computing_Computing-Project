"""Shared pytest fixtures: common parameter sets, basis states and tolerances."""

import pytest

from rydberg.units import mhz_to_angular

# Parameter sets (omega, delta, phi, v) in rad/us and rad. They include a
# non-zero phase, because some sign errors only show up when phi != 0.
PARAMETER_SETS = {
    "resonant_no_interaction": (mhz_to_angular(1.0), 0.0, 0.0, 0.0),
    "resonant_blockade": (mhz_to_angular(1.0), 0.0, 0.0, mhz_to_angular(100.0)),
    "detuned_with_phase": (
        mhz_to_angular(1.0),
        mhz_to_angular(3.0),
        0.7,
        mhz_to_angular(20.0),
    ),
    "negative_detuning": (
        mhz_to_angular(2.5),
        mhz_to_angular(-50.0),
        -1.9,
        mhz_to_angular(5.0),
    ),
}


@pytest.fixture(params=list(PARAMETER_SETS.values()), ids=list(PARAMETER_SETS))
def params(request: pytest.FixtureRequest) -> tuple[float, float, float, float]:
    """Return a parameter set (omega, delta, phi, v)."""
    result: tuple[float, float, float, float] = request.param
    return result
