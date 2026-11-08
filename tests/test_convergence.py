"""Tests for the convergence utilities on synthetic data with known answers."""

import numpy as np
import pytest

from rydberg.convergence import (
    agrees_to_sig_figs,
    fit_power_law,
    largest_passing_step,
    max_abs_error,
    min_passing_steps,
    sig_fig_tolerance,
    subsample_to_grid,
)


@pytest.mark.parametrize(("p", "a"), [(1.0, 3.0), (2.0, 0.5), (-1.5, 7.0), (4.0, 1e-3)])
def test_fit_power_law_recovers_exact_exponent(p: float, a: float) -> None:
    x = np.logspace(-5, -1, 9)
    fitted_p, fitted_a = fit_power_law(x, a * x**p)
    assert fitted_p == pytest.approx(p, rel=1e-10)
    assert fitted_a == pytest.approx(a, rel=1e-8)


def test_fit_power_law_with_noise() -> None:
    rng = np.random.default_rng(20261108)
    x = np.logspace(-4, -1, 30)
    y = 2.0 * x**2 * np.exp(rng.normal(scale=0.05, size=x.size))
    p, _ = fit_power_law(x, y)
    assert p == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([1.0], [1.0]),
        ([1.0, 2.0], [1.0]),
        ([1.0, -2.0], [1.0, 2.0]),
        ([1.0, 2.0], [0.0, 2.0]),
        ([1.0, np.nan], [1.0, 2.0]),
    ],
)
def test_fit_power_law_rejects_bad_input(x: list[float], y: list[float]) -> None:
    with pytest.raises(ValueError):
        fit_power_law(x, y)


def test_sig_fig_tolerance() -> None:
    assert sig_fig_tolerance(0.9985, 4) == pytest.approx(5e-5 * 0.9985)
    assert sig_fig_tolerance(-2.0, 2) == pytest.approx(1e-2)
    with pytest.raises(ValueError):
        sig_fig_tolerance(1.0, 0)


def test_agrees_to_sig_figs() -> None:
    reference = 0.998543
    assert agrees_to_sig_figs(reference + 4.9e-5, reference)
    assert not agrees_to_sig_figs(reference + 5.1e-5, reference)
    assert agrees_to_sig_figs(reference - 4.9e-5, reference)
    assert not agrees_to_sig_figs(0.9980, reference)


def test_max_abs_error() -> None:
    assert max_abs_error([1.0, 2.0, 3.0], [1.0, 2.5, 2.9]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Shape"):
        max_abs_error([1.0], [1.0, 2.0])


def test_subsample_to_grid() -> None:
    fine = np.linspace(0.0, 1.0, 1001)
    values = np.stack([fine, fine**2], axis=-1)
    coarse = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(
        subsample_to_grid(fine, values, coarse), np.stack([coarse, coarse**2], axis=-1)
    )
    with pytest.raises(ValueError, match="coincide"):
        subsample_to_grid(fine, values, np.array([0.0005]))
    with pytest.raises(ValueError, match="outside"):
        subsample_to_grid(fine, values, np.array([1.5]))


def test_largest_passing_step_simple() -> None:
    steps = np.array([1e-1, 1e-2, 1e-3, 1e-4])
    errors = 0.5 * steps  # passes for step <= 2e-3
    assert largest_passing_step(steps, errors, 1e-3) == pytest.approx(1e-3)
    # Order of the input does not matter.
    assert largest_passing_step(steps[::-1], errors[::-1], 1e-3) == pytest.approx(1e-3)


def test_largest_passing_step_ignores_accidental_pass() -> None:
    """A coarse step that passes by luck is rejected if a finer one fails."""
    steps = np.array([1e-1, 5e-2, 1e-2, 1e-3])
    errors = np.array([1e-6, 1e-1, 1e-2, 1e-5])
    assert largest_passing_step(steps, errors, 1e-4) == pytest.approx(1e-3)


def test_largest_passing_step_none_pass() -> None:
    assert largest_passing_step([1.0, 0.1], [1.0, 1.0], 1e-3) is None


def test_min_passing_steps_finds_threshold() -> None:
    # Error 1/n, tolerance 1/37 -> smallest passing n is 37.
    calls: list[int] = []

    def error(n: int) -> float:
        calls.append(n)
        return 1.0 / n

    assert min_passing_steps(error, 2, 1000, 1.0 / 37) == 37
    assert len(calls) < 20  # bisection, not a linear scan


def test_min_passing_steps_validates_brackets() -> None:
    with pytest.raises(ValueError):
        min_passing_steps(lambda n: 1.0 / n, 10, 5, 0.01)
    with pytest.raises(ValueError, match="already"):
        min_passing_steps(lambda n: 1.0 / n, 200, 1000, 0.01)
    with pytest.raises(ValueError, match="does not"):
        min_passing_steps(lambda n: 1.0 / n, 2, 50, 0.01)
