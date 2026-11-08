"""Tests for the Trotter-Suzuki propagator of handbook Eq. (5.7)."""

from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from rydberg.convergence import fit_power_law, max_abs_error, subsample_to_grid
from rydberg.evolution.eigen import evolve
from rydberg.evolution.trotter import (
    evolve_trotter,
    evolve_trotter_reference,
    n_steps_for,
)
from rydberg.hamiltonians import single_atom_hamiltonian, two_atom_hamiltonian
from rydberg.protocols import (
    cubic_detuning,
    detuning_sweep_hamiltonian,
    landau_zener_bell_population,
    linear_detuning,
)
from rydberg.states import KET_00, PSI_PLUS, norm, overlap_mod_squared
from rydberg.units import mhz_to_angular

Params = tuple[float, float, float, float]
Sample = Literal["left", "midpoint"]
Hamiltonian = Callable[[NDArray[np.float64] | float], NDArray[np.complex128]]

OMEGA = mhz_to_angular(1.0)
DELTA_MAX = mhz_to_angular(50.0)
V_BLOCKADE = mhz_to_angular(100.0)
T_MAX = 1.0
"""Handbook Fig. 5.2 parameters: rad/us and us."""

ROUNDOFF = 1e-12
"""Tolerance for results that are exact up to accumulated round-off."""


def cubic_sweep(v: float = V_BLOCKADE) -> Hamiltonian:
    """Return H(t) for the cubic sweep of handbook Fig. 5.2."""
    return detuning_sweep_hamiltonian(
        OMEGA, lambda t: cubic_detuning(t, DELTA_MAX, T_MAX), 0.0, v
    )


def populations(states: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return P(|00>) and P(|Psi+>) along a trajectory, shape (n_times, 2)."""
    return np.stack(
        [overlap_mod_squared(states, KET_00), overlap_mod_squared(states, PSI_PLUS)],
        axis=-1,
    )


# --- n_steps_for and argument checking ----------------------------------------


def test_n_steps_for_exact_division() -> None:
    assert n_steps_for(1.0, 1e-3) == 1000
    assert n_steps_for(1.0, 0.1) == 10  # 1.0 / 0.1 = 9.999999999999998
    assert n_steps_for(2.5, 0.5, t0=0.5) == 4


@pytest.mark.parametrize("dt", [0.0, -1e-3])
def test_rejects_non_positive_dt(dt: float) -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        evolve_trotter(cubic_sweep(), KET_00, T_MAX, dt)


def test_rejects_non_dividing_dt() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        evolve_trotter(cubic_sweep(), KET_00, T_MAX, 0.3)


def test_rejects_empty_interval() -> None:
    with pytest.raises(ValueError, match="greater than t0"):
        evolve_trotter(cubic_sweep(), KET_00, 0.0, 0.1)


def test_rejects_unknown_sample() -> None:
    with pytest.raises(ValueError, match="sample"):
        evolve_trotter(cubic_sweep(), KET_00, T_MAX, 0.1, sample="right")  # type: ignore[arg-type]


def test_rejects_incompatible_state() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        evolve_trotter(cubic_sweep(), np.array([1.0, 0.0]), T_MAX, 0.1, vectorised=True)


def test_rejects_non_hermitian() -> None:
    def h(t: float) -> NDArray[np.complex128]:
        return np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="Hermitian"):
        evolve_trotter(h, np.array([1.0, 0.0]), T_MAX, 0.1)


# --- exactness for constant H and internal consistency ------------------------


@pytest.mark.parametrize("sample", ["left", "midpoint"])
@pytest.mark.parametrize("n_steps", [1, 7, 250])
def test_constant_hamiltonian_matches_eigen_evolve(
    params: Params, sample: Sample, n_steps: int
) -> None:
    """For constant H each step is exact, so any dt reproduces Eq. (5.6)."""
    h = two_atom_hamiltonian(*params)
    t_max = 2.3
    times, states = evolve_trotter(
        lambda t: h, KET_00, t_max, t_max / n_steps, sample=sample  # type: ignore[arg-type]
    )
    assert times.shape == (n_steps + 1,)
    assert states.shape == (n_steps + 1, 4)
    assert times[0] == 0.0
    assert times[-1] == t_max
    np.testing.assert_allclose(states, evolve(h, KET_00, times), atol=ROUNDOFF)


def test_initial_time_offset() -> None:
    """Starting at t0 samples H(t) on [t0, t_max], not [0, t_max - t0]."""
    h = cubic_sweep()
    times, states = evolve_trotter(h, KET_00, T_MAX, 1e-3, t0=0.4, vectorised=True)
    np.testing.assert_allclose(times[0], 0.4)
    _, full = evolve_trotter(h, KET_00, T_MAX, 1e-3, vectorised=True)
    # From |00> at t0 = 0.4 is not the same as continuing the full trajectory.
    assert not np.allclose(states[-1], full[-1])
    _, ref = evolve_trotter_reference(h, KET_00, T_MAX, 1e-3, t0=0.4)
    np.testing.assert_allclose(states, ref, atol=ROUNDOFF)


@pytest.mark.parametrize("sample", ["left", "midpoint"])
def test_batched_matches_literal_implementation(sample: Sample) -> None:
    """The chunked, vectorised loop equals one eigen.propagator call per step."""
    h = cubic_sweep()
    n = 5000  # more than one chunk
    _, fast = evolve_trotter(h, KET_00, T_MAX, T_MAX / n, sample=sample, vectorised=True)
    _, scalar = evolve_trotter(h, KET_00, T_MAX, T_MAX / n, sample=sample)
    _, literal = evolve_trotter_reference(h, KET_00, T_MAX, T_MAX / n, sample=sample)
    np.testing.assert_allclose(fast, literal, atol=ROUNDOFF)
    np.testing.assert_allclose(scalar, literal, atol=ROUNDOFF)


def test_left_and_midpoint_sample_different_times() -> None:
    """Sanity check that the sampling choice is actually applied."""
    h = cubic_sweep()
    _, left = evolve_trotter(h, KET_00, T_MAX, 0.01, sample="left", vectorised=True)
    _, mid = evolve_trotter(h, KET_00, T_MAX, 0.01, sample="midpoint", vectorised=True)
    assert max_abs_error(left[-1], mid[-1]) > 1e-3


@pytest.mark.parametrize("sample", ["left", "midpoint"])
@pytest.mark.parametrize("v", [0.0, V_BLOCKADE])
def test_norm_conserved_for_sweep(sample: Sample, v: float) -> None:
    """Every step is unitary, so the norm is conserved even for a coarse dt."""
    _, states = evolve_trotter(
        cubic_sweep(v), KET_00, T_MAX, 1e-3, sample=sample, vectorised=True
    )
    np.testing.assert_allclose(norm(states), 1.0, atol=ROUNDOFF)


# --- convergence and independent references ------------------------------------

STEP_COUNTS = np.array([1000, 2000, 4000, 8000])
"""Step counts for the order tests, all in the asymptotic regime (dt <= 1e-3 us)."""

REFERENCE_STEPS = 128_000
"""Midpoint reference; its error (~5e-10) is < 1% of the smallest error fitted."""


@pytest.fixture(scope="module")
def reference_populations() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fine midpoint run of the cubic sweep at V/2pi = 100 MHz: (times, P)."""
    times, states = evolve_trotter(
        cubic_sweep(), KET_00, T_MAX, T_MAX / REFERENCE_STEPS,
        sample="midpoint", vectorised=True,
    )
    return times, populations(states)


@pytest.mark.parametrize("sample", ["left", "midpoint"])
def test_no_interaction_sweep_is_product_of_single_atom_runs(sample: Sample) -> None:
    """At V = 0, H(t) = h(t) (x) 1 + 1 (x) h(t), so each step factorises exactly.

    The two-atom Trotter state must then equal the Kronecker product of two
    independent single-atom Trotter runs at every step, for any dt and either
    sampling rule. This checks the time-dependent path (indexing of H(t),
    sampling times) against an independent construction, and implies the
    product-state identity P_Psi+ = 2 sqrt(P_00) (1 - sqrt(P_00)).
    """
    dt = 1e-3

    def single(t: NDArray[np.float64] | float) -> NDArray[np.complex128]:
        return single_atom_hamiltonian(OMEGA, float(cubic_detuning(t, DELTA_MAX, T_MAX)))

    ket_0 = np.array([1.0, 0.0], dtype=np.complex128)
    _, one = evolve_trotter(single, ket_0, T_MAX, dt, sample=sample)
    _, two = evolve_trotter(cubic_sweep(0.0), KET_00, T_MAX, dt, sample=sample)
    product = np.einsum("ti,tj->tij", one, one).reshape(-1, 4)
    np.testing.assert_allclose(two, product, atol=ROUNDOFF)

    p_00 = overlap_mod_squared(two, KET_00)
    p_plus = overlap_mod_squared(two, PSI_PLUS)
    np.testing.assert_allclose(p_plus, 2 * np.sqrt(p_00) * (1 - np.sqrt(p_00)), atol=ROUNDOFF)


@pytest.mark.convergence
@pytest.mark.parametrize(
    ("sample", "expected_order"), [("left", 1.0), ("midpoint", 2.0)]
)
def test_convergence_order(
    reference_populations: tuple[NDArray[np.float64], NDArray[np.float64]],
    sample: Sample,
    expected_order: float,
) -> None:
    """Max population error over the trajectory scales as dt (left), dt^2 (midpoint).

    Measured slopes over these step counts are 1.02 (left) and 2.00
    (midpoint); the tolerance of 0.1 allows for pre-asymptotic curvature.
    """
    ref_times, ref_pops = reference_populations
    errors = []
    for n in STEP_COUNTS:
        times, states = evolve_trotter(
            cubic_sweep(), KET_00, T_MAX, T_MAX / n, sample=sample, vectorised=True
        )
        expected = subsample_to_grid(ref_times, ref_pops, times)
        errors.append(max_abs_error(populations(states), expected))
    order, _ = fit_power_law(T_MAX / STEP_COUNTS, errors)
    assert order == pytest.approx(expected_order, abs=0.1)


@pytest.mark.convergence
def test_midpoint_far_more_accurate_than_left(
    reference_populations: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    ref_times, ref_pops = reference_populations
    errors = {}
    for sample in ("left", "midpoint"):
        times, states = evolve_trotter(
            cubic_sweep(), KET_00, T_MAX, 1e-3, sample=sample, vectorised=True
        )
        expected = subsample_to_grid(ref_times, ref_pops, times)
        errors[sample] = max_abs_error(populations(states), expected)
    assert errors["midpoint"] < 1e-2 * errors["left"]


@pytest.mark.parametrize("v", [0.0, V_BLOCKADE])
def test_agrees_with_solve_ivp(v: float) -> None:
    """Independent check against an adaptive Runge-Kutta solution of Eq. (5.5).

    DOP853 at rtol = atol = 1e-12 is accurate to ~1e-10 in the populations.
    The midpoint rule at dt = 5e-5 us has a measured max population error of
    1.1e-8 (V = 0) and 7.0e-9 (V/2pi = 100 MHz), consistent with ~3 dt^2, so
    1e-7 is a safe bound that still fails for any first-order error (the left
    rule at this dt is off by ~1e-4).
    """
    h = cubic_sweep(v)
    times, states = evolve_trotter(
        h, KET_00, T_MAX, 5e-5, sample="midpoint", vectorised=True
    )
    t_eval = times[::200]
    sol = solve_ivp(
        lambda t, y: -1j * (h(t) @ y),
        (0.0, T_MAX),
        KET_00.astype(np.complex128),
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-12,
    )
    assert sol.success
    assert max_abs_error(populations(states[::200]), populations(sol.y.T)) < 1e-7


@pytest.mark.slow
@pytest.mark.parametrize("t_max", [1.0, 3.0, 5.0])
def test_linear_sweep_follows_landau_zener(t_max: float) -> None:
    """Deep in the blockade regime the linear sweep is a Landau-Zener crossing.

    The finite sweep range +-Delta_max makes the exact result oscillate about
    the Landau-Zener value; the measured deviations are 0.013, 0.005 and
    -0.009 for t_max = 1, 3 and 5 us.
    """
    v = mhz_to_angular(10_000.0)
    h = detuning_sweep_hamiltonian(
        OMEGA, lambda t: linear_detuning(t, DELTA_MAX, t_max), 0.0, v
    )
    _, states = evolve_trotter(
        h, KET_00, t_max, t_max / round(t_max / 1e-5), sample="midpoint", vectorised=True
    )
    p_bell = overlap_mod_squared(states[-1], PSI_PLUS)
    expected = landau_zener_bell_population(OMEGA, DELTA_MAX, t_max)
    assert p_bell == pytest.approx(float(expected), abs=0.03)
