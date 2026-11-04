"""Shared plotting style and figure export.

All presented figures go through :func:`use_style` and :func:`save_figure` so
that the report, poster and slides share consistent fonts, sizes and colours.
"""

from importlib.resources import as_file, files
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

DEFAULT_FIGURE_DIR = Path("results/figures")


def use_style(name: str = "publication") -> None:
    """Activate a Matplotlib style sheet shipped with the package.

    Parameters
    ----------
    name : str, default "publication"
        Name of a ``.mplstyle`` file in ``rydberg/styles/`` (without suffix).

    Raises
    ------
    FileNotFoundError
        If no style sheet with that name exists.
    """
    resource = files("rydberg").joinpath("styles", f"{name}.mplstyle")
    with as_file(resource) as style_path:
        if not style_path.is_file():
            raise FileNotFoundError(f"Style sheet not found: {name}.mplstyle")
        plt.style.use(style_path)


def save_figure(
    fig: Figure,
    name: str,
    outdir: str | Path = DEFAULT_FIGURE_DIR,
) -> tuple[Path, Path]:
    """Save a figure as both PDF (for LaTeX) and PNG (for the README).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        File stem, without suffix.
    outdir : str or pathlib.Path, default "results/figures"
        Output directory; created if it does not exist.

    Returns
    -------
    tuple of pathlib.Path
        Paths of the written PDF and PNG files.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{name}.pdf"
    png_path = outdir / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    return pdf_path, png_path
