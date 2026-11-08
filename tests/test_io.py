"""Tests for saving and loading results with metadata."""

from pathlib import Path

import numpy as np
import pytest

from rydberg import __version__
from rydberg.io import load_results, save_results


def test_round_trip_preserves_arrays_and_metadata(tmp_path: Path) -> None:
    t = np.linspace(0.0, 1.0, 11)
    psi = np.exp(-1j * t)
    npz_path, json_path = save_results(
        tmp_path / "run", {"t": t, "psi": psi}, {"omega": 2 * np.pi}
    )
    assert npz_path.name == "run.npz"
    assert json_path.name == "run.json"

    arrays, metadata = load_results(tmp_path / "run")
    np.testing.assert_array_equal(arrays["t"], t)
    np.testing.assert_array_equal(arrays["psi"], psi)
    assert arrays["psi"].dtype == np.complex128
    assert metadata["omega"] == pytest.approx(2 * np.pi)


def test_provenance_metadata_is_added(tmp_path: Path) -> None:
    save_results(tmp_path / "run", {"x": [1, 2, 3]})
    _, metadata = load_results(tmp_path / "run.npz")
    assert metadata["package_version"] == __version__
    assert isinstance(metadata["git_commit"], str)
    assert metadata["git_commit"]
    assert metadata["git_dirty"] in (True, False, None)
    assert metadata["timestamp_utc"].endswith("+00:00")


def test_suffix_is_normalised_and_dirs_created(tmp_path: Path) -> None:
    npz_path, json_path = save_results(tmp_path / "nested" / "run.json", {"x": [0]})
    assert npz_path == tmp_path / "nested" / "run.npz"
    assert json_path.is_file()


def test_load_missing_results_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Results file not found"):
        load_results(tmp_path / "missing")
