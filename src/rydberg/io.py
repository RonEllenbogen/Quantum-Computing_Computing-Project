"""Saving and loading of numerical results with provenance metadata.

Arrays are stored in a compressed ``.npz`` file with a sidecar ``.json`` file
holding the run metadata, the git commit, a UTC timestamp and the package version.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from rydberg import __version__


def _git_commit() -> str:
    """Return the current git commit hash, or ``"unknown"`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_dirty() -> bool | None:
    """Return True if tracked files have uncommitted changes, None if unknown.

    A result produced from a dirty tree cannot be reproduced from
    ``git_commit`` alone, so this flag is stored alongside it. Generated
    outputs under ``results/`` are ignored: regenerating figures rewrites
    tracked files there, which says nothing about the code that made them.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                ":(top)",
                ":(top,exclude)results",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _paths(path: str | Path) -> tuple[Path, Path]:
    """Return the ``.npz`` and ``.json`` paths sharing the stem of ``path``."""
    base = Path(path)
    if base.suffix in {".npz", ".json"}:
        base = base.with_suffix("")
    return base.with_name(base.name + ".npz"), base.with_name(base.name + ".json")


def save_results(
    path: str | Path,
    arrays: dict[str, npt.ArrayLike],
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save arrays to ``.npz`` and metadata to a sidecar ``.json`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output path. Any ``.npz``/``.json`` suffix is replaced, so
        ``results/data/run`` writes ``run.npz`` and ``run.json``.
    arrays : dict of str to array_like
        Named arrays to store.
    metadata : dict, optional
        JSON-serialisable run metadata (e.g. the config used). The keys
        ``git_commit``, ``git_dirty`` (True if tracked files had uncommitted
        changes), ``timestamp_utc`` and ``package_version`` are added
        automatically.

    Returns
    -------
    tuple of pathlib.Path
        Paths of the written ``.npz`` and ``.json`` files.
    """
    npz_path, json_path = _paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(metadata or {})
    meta.update(
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        package_version=__version__,
    )

    stored: dict[str, Any] = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(npz_path, **stored)
    json_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return npz_path, json_path


def load_results(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load results written by :func:`save_results`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.npz`` file, the ``.json`` file or their common stem.

    Returns
    -------
    arrays : dict of str to numpy.ndarray
        The stored arrays.
    metadata : dict
        The stored metadata.

    Raises
    ------
    FileNotFoundError
        If either the ``.npz`` or the ``.json`` file is missing.
    """
    npz_path, json_path = _paths(path)
    for p in (npz_path, json_path):
        if not p.is_file():
            raise FileNotFoundError(f"Results file not found: {p.resolve()}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    metadata = json.loads(json_path.read_text())
    return arrays, metadata
