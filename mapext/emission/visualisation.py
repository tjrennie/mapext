"""Provide visualization tools for polarised emission models.

Include functions to extract fittable polarised models, format axes for
Stokes parameters, and plot various visualizations such as IPA and IQU plots.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from mapext.core.stokes import StokesComp, display_parameters
from mapext.emission.core import (
    compoundFittablePolarisedEmissionModel,
    fittablePolarisedEmissionModel,
)

logger = logging.getLogger(__name__)


def extract_fittable_polarised_models(model):
    """Recursively extract all instances of FittablePolarisedEmissionModel from a compound model."""
    found_models = []

    if isinstance(model, fittablePolarisedEmissionModel):
        found_models.append(model)

    if isinstance(model, compoundFittablePolarisedEmissionModel):
        found_models.extend(extract_fittable_polarised_models(model.left))
        found_models.extend(extract_fittable_polarised_models(model.right))

    return found_models


def format_angle(value, tick_number):
    """Format a given angle value as a multiple of π.

    Parameters
    ----------
    value : float
        The angle value to format.
    tick_number : int
        The tick number (unused in this function).

    Returns
    -------
    str
        The formatted angle as a string, e.g., "0.5π".
    """
    fraction = np.round(value / np.pi, 2)  # Convert to multiple of π
    return f"{fraction}" + r"$\pi$"  # Format as "0.5π"


def format_stokes_axis(ax, stokes, grid=True):
    """Format the axis for a given Stokes parameter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to format.
    stokes : StokesComp
        The Stokes parameter to format the axis for.
    grid : bool, optional
        Whether to display a grid on the axis (default is True).
    """
    if int(stokes) <= 4:
        ax.set_yscale("log")

    if int(stokes) == int(StokesComp("A")):
        ax.set_ylim(-np.pi / 2, np.pi / 2)
        ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_angle))

        ax2 = ax.twinx()
        ax2.yaxis.set_major_locator(plt.MultipleLocator(45))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(15))
        ax2.set_ylim(-90, 90)

    ax.set_ylabel(display_parameters[stokes.letter])
    if grid:
        ax.grid("both", "minor", c="k", alpha=0.2)
        ax.grid("both", "major", c="k", alpha=0.4)


def IPA_plot(model, *args):
    """Plot the Stokes I, P, and A parameters for a given model.

    Parameters
    ----------
    model : callable
        The polarised emission model to plot.
    *args : tuple
        Additional arguments passed to the model, typically including
        frequency values and other required parameters.
    """
    fig, axs = plt.subplots(3, sharex=True)
    plt.subplots_adjust(hspace=0, top=0.95, bottom=0.15)

    axs[0].set_xscale("log")
    axs[0].set_xlim(np.nanmin(args[0]), np.nanmax(args[0]))
    axs[-1].set_xlabel(r"$\nu$ [GHz]")

    stokesobj = [StokesComp(_) for _ in ["I", "P", "A"]]

    submodels = extract_fittable_polarised_models(model)

    for ax, stokes in zip(axs, stokesobj):
        format_stokes_axis(ax, stokes)

        ax.plot(args[0], model(*args, stokes), c="k", zorder=3)
        if int(stokes) in [
            int(StokesComp("Q")),
            int(StokesComp("U")),
            int(StokesComp("V")),
        ]:
            ax.plot(args[0], -model(*args, stokes), c="k", zorder=3, ls="dashed")

        if int(stokes) != int(StokesComp("A")):
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()

        for m in submodels:
            line = ax.plot(args[0], m(*args, stokes), label=m.name, alpha=0.8)
            if int(stokes) in [
                int(StokesComp("Q")),
                int(StokesComp("U")),
                int(StokesComp("V")),
            ]:
                ax.plot(
                    args[0],
                    -m(*args, stokes),
                    c=line[0].get_color(),
                    alpha=line[0].get_alpha(),
                    ls="dashed",
                )

        if int(stokes) != int(StokesComp("A")):
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

    axs[-1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=5,
        frameon=False,
        facecolor=(1, 1, 1, 0.1),
    )


def IQU_plot(model, *args):
    """Plot the Stokes I, Q, and U parameters for a given model.

    Parameters
    ----------
    model : callable
        The polarised emission model to plot.
    *args : tuple
        Additional arguments passed to the model, typically including
        frequency values and other required parameters.
    """
    fig, axs = plt.subplots(3, sharex=True)
    plt.subplots_adjust(hspace=0, top=0.95, bottom=0.15)

    axs[0].set_xscale("log")
    axs[0].set_xlim(np.nanmin(args[0]), np.nanmax(args[0]))
    axs[-1].set_xlabel(r"$\nu$ [GHz]")

    stokesobj = [StokesComp(_) for _ in ["I", "Q", "U"]]

    submodels = extract_fittable_polarised_models(model)

    for ax, stokes in zip(axs, stokesobj):
        format_stokes_axis(ax, stokes)

        ax.plot(args[0], model(*args, stokes), c="k", zorder=3)
        if int(stokes) in [
            int(StokesComp("Q")),
            int(StokesComp("U")),
            int(StokesComp("V")),
        ]:
            ax.plot(args[0], -model(*args, stokes), c="k", zorder=3, ls="dashed")

        if int(stokes) != int(StokesComp("A")):
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()

        for m in submodels:
            line = ax.plot(args[0], m(*args, stokes), label=m.name, alpha=0.8)
            if int(stokes) in [
                int(StokesComp("Q")),
                int(StokesComp("U")),
                int(StokesComp("V")),
            ]:
                ax.plot(
                    args[0],
                    -m(*args, stokes),
                    c=line[0].get_color(),
                    alpha=line[0].get_alpha(),
                    ls="dashed",
                )

        if int(stokes) != int(StokesComp("A")):
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

    axs[-1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=5,
        frameon=False,
        facecolor=(1, 1, 1, 0.1),
    )
