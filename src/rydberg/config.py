"""Loading of TOML run configurations.

Every script in ``scripts/`` reads its parameters from a TOML file in
``configs/``, so that no physical parameters are hard-coded in the scripts.
"""

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.toml`` file.

    Returns
    -------
    dict
        Parsed configuration as a (possibly nested) dictionary.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist or is not a file.
    ValueError
        If the file is not valid TOML.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as err:
        raise ValueError(f"Invalid TOML in config file {path}: {err}") from err
