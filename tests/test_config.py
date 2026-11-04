"""Tests for the TOML configuration loader."""

from pathlib import Path

import pytest

from rydberg.config import load_config


def test_load_config_parses_nested_tables(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        'name = "test"\n\n[laser]\nomega_mhz = 1.0\ndetunings = [0.0, 5.0, 10.0]\n'
    )
    config = load_config(path)
    assert config["name"] == "test"
    assert config["laser"]["omega_mhz"] == pytest.approx(1.0)
    assert config["laser"]["detunings"] == [0.0, 5.0, 10.0]


def test_load_config_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text("x = 1\n")
    assert load_config(str(path)) == {"x": 1}


def test_load_config_empty_file_gives_empty_dict(tmp_path: Path) -> None:
    path = tmp_path / "empty.toml"
    path.write_text("# comment only\n")
    assert load_config(path) == {}


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "does_not_exist.toml")


def test_load_config_invalid_toml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("this is = = not toml\n")
    with pytest.raises(ValueError, match="Invalid TOML"):
        load_config(path)
