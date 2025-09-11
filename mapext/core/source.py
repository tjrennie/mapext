"""Module defining the astroSrc class for representing astronomical sources and their associated data."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from mapext.core.stokes import queryable_parameters

logger = logging.getLogger(__name__)

__all__ = ["astroSrc"]


class astroSrc:
    """Class to hold information pertaining to a specific astronomical source.

    This class is currently aimed at point sources, although will be expanded to be more flexible.
    """

    def __init__(self, name="unnamed", coords=None, frame="galactic"):
        """Initialization function.

        Parameters
        ----------
        name : str, optional
            Source name.
        coords : list or SkyCoord
            Center coordinates for the source. If a list is provided, it should be [lon, lat] in degrees.
        frame : str, optional
            Coordinate frame if a list of floats is supplied for coords (default 'galactic').
        """
        self.name = name

        if coords is None:
            raise ValueError(
                "Coordinates must be provided as a list or a SkyCoord object."
            )

        if isinstance(coords, SkyCoord):
            self.coord = coords
            self.frame = coords.frame.name
        else:
            self.coord = SkyCoord(*coords, frame=frame, unit="degree")
            self.frame = frame

        self.flux = np.array(
            [],
            dtype=[
                ("name", "<U40"),  # Name or label for the flux measurement
                ("freq", "float"),  # Frequency in Hz
                ("bandwidth", "float"),  # Bandwidth in Hz
                (
                    "values",
                    "float",
                    (7,),
                ),  # Stokes parameters I, Q, U, V, P, A, PF in Jy, Jy, Jy, Jy, Jy, degrees, percent
                ("errors", "float", (7,)),  # Errors in units as applicable
                ("epoch", "float64"),  # Epoch in decimal years
            ],
        )

    def __repr__(self):
        return f"<astroSrc: {self.name}, Coord: {self.coord.to_string('decimal')}, Frame: {self.frame}, Flux entries: {len(self.flux)}>"

    # ==========================================================================
    # Flux Measurement Management

    def add_flux(self, name, freq, bandwidth, values, errors, epoch=None):
        """Add a flux measurement entry.

        Parameters
        ----------
        name : str
            Name or label for this flux measurement.
        freq : float
            Frequency in Hz or relevant units.
        bandwidth : float
            Bandwidth in Hz or relevant units.
        values : array-like of float or dictionary of str: float
            Flux values either as an array of 7 elements or a dictionary with keys corresponding to Stokes parameters (I, Q, U, V, P, A, PF).
        errors : array-like of float or dictionary of str: float
            Errors associated with the flux values.
        epoch : float, str, or astropy.time.Time, optional
            Observation time, stored as decimal year.
        """
        if isinstance(values, dict):
            values = np.array(
                [
                    values.get(param, np.nan)
                    for param in ["I", "Q", "U", "V", "P", "A", "PF"]
                ]
            )
        if isinstance(errors, dict):
            errors = np.array(
                [
                    errors.get(param, np.nan)
                    for param in ["I", "Q", "U", "V", "P", "A", "PF"]
                ]
            )

        if isinstance(epoch, Time):
            epoch_year = epoch.decimalyear
        elif isinstance(epoch, str):
            epoch_year = Time(epoch).decimalyear
        elif isinstance(epoch, (float, int)):
            epoch_year = float(epoch)
        elif epoch is None:
            epoch_year = np.nan
        else:
            raise ValueError(
                "Epoch must be a float, string, or astropy.time.Time instance."
            )

        new_entry = np.array(
            [(name, freq, bandwidth, values, errors, epoch_year)], dtype=self.flux.dtype
        )
        self.flux = np.append(self.flux, new_entry)

    def plot_stokesflux(
        self, stokes, ax=None, c="black", marker="x", label_axes=True, epoch=None
    ):
        """Plot the Stokes parameters of the flux measurements.

        Parameters
        ----------
        stokes : str
            The Stokes parameter to plot (e.g., 'I', 'Q', 'U', 'V').
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses the current axes.
        c : str, optional
            Color of the markers (default 'black').
        marker : str, optional
            Marker style for the plot (default 'x').
        label_axes : bool, optional
            Whether to label the axes (default True).
        epoch : float or str, optional
            Observation time as decimal year.
        """
        if stokes not in queryable_parameters:
            raise ValueError(
                f"Invalid Stokes parameter: {stokes}. Must be one of {queryable_parameters}."
            )
        stokes_index = queryable_parameters.index(stokes)

        if ax is None:
            ax = plt.gca()

        secvar_correction = 1.0
        if epoch:
            if hasattr(self, "model_secvar"):
                secvar_correction = self.model_secvar(
                    **{name: self.flux[name] for name in self.flux.dtype.names}
                )
            else:
                logger.warning("No secvar model available, skipping secvar correction.")

        ax.errorbar(
            self.flux["freq"],
            self.flux["values"][:, stokes_index] * secvar_correction,
            xerr=self.flux["bandwidth"] / 2,
            yerr=self.flux["errors"][:, stokes_index] * secvar_correction,
            ls="none",
            marker=marker,
            c=c,
        )

        if label_axes:
            ax.set_xlabel(r"$\nu$ [Hz]")
            stokes_upper = stokes.upper()

            subscript = rf"\nu, {epoch}" if epoch is not None else r"\nu"

            if stokes_upper in ["I", "Q", "U", "V"]:
                ax.set_ylabel(rf"$S^{{({stokes_upper})}}_{{{subscript}}}$ [Jy]")
            elif stokes_upper == "P":
                ax.set_ylabel(rf"$P_{{{subscript}}}$ [%]")  # polarized intensity
            elif stokes_upper == "A":
                ax.set_ylabel(rf"$\phi_{{{subscript}}}$ [deg]")  # polarization angle
            elif stokes_upper == "PF":
                ax.set_ylabel(
                    rf"$\frac{{P_{{{subscript}}}}}{{S^{{(I)}}_{{{subscript}}}}}$ [%]"
                )  # polarization fraction
            else:
                raise ValueError(f"Unknown Stokes parameter: {stokes_upper}")
