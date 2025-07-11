"""Module defining the astroSrc class for representing astronomical sources and their associated data."""

import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.time import Time

from mapext.core.stokes import queryable_parameters

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
                ("name", "<U40"),
                ("freq", "float"),
                ("bandwidth", "float"),
                ("values", "float", (7,)),
                ("errors", "float", (7,)),
                ("epoch", "float64"),  # Epoch in decimal years
            ],
        )
    
    @classmethod
    def from_csv(cls, filename):
        """Load sources from a CSV file.

        The CSV should contain at least columns: name, lon, lat, frame (optional).
        Additional columns for flux can be added depending on your structure.

        Returns
        -------
        list of astroSrc
        """
        sources = []
        with open(filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row["name"]
                coords = [float(row["lon"]), float(row["lat"])]
                frame = row.get("frame", "galactic")

                src = cls(name=name, coords=coords, frame=frame)

                # Optional: parse flux if provided
                if "freq" in row and "bandwidth" in row and "epoch" in row:
                    freq = float(row["freq"])
                    bandwidth = float(row["bandwidth"])
                    epoch = float(row["epoch"])
                    values = {
                        key: float(row[key])
                        for key in ["I", "Q", "U", "V", "P", "A", "PF"]
                        if key in row
                    }
                    errors = {
                        f"{key}_err": float(row[f"{key}_err"])
                        for key in ["I", "Q", "U", "V", "P", "A", "PF"]
                        if f"{key}_err" in row
                    }
                    # Strip "_err" keys to match expected input
                    errors = {k.replace("_err", ""): v for k, v in errors.items()}
                    src.add_flux(
                        name="flux_entry",
                        freq=freq,
                        bandwidth=bandwidth,
                        values=values,
                        errors=errors,
                        epoch=epoch,
                    )

                sources.append(src)
        return sources

    @classmethod
    def from_json(cls, filename):
        """Load sources from a JSON file.

        Expects a list of dicts, each with keys: name, coords (list), frame (optional), flux (optional).

        Returns
        -------
        list of astroSrc
        """
        with open(filename) as f:
            data = json.load(f)

        sources = []
        for item in data:
            name = item["name"]
            coords = item["coords"]
            frame = item.get("frame", "galactic")

            src = cls(name=name, coords=coords, frame=frame)

            if "flux" in item:
                for flux_entry in item["flux"]:
                    src.add_flux(
                        flux_entry["name"],
                        flux_entry["freq"],
                        flux_entry["bandwidth"],
                        flux_entry["values"],
                        flux_entry["errors"],
                        flux_entry.get("epoch", None),
                    )

            sources.append(src)
        return sources

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

    def plot_stokesflux(self, stokes, ax=None, c='black', marker='x', label_axes=True):
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
        """
        if stokes not in queryable_parameters:
            raise ValueError(f"Invalid Stokes parameter: {stokes}. Must be one of {queryable_parameters}.")
        stokes_index = queryable_parameters.index(stokes)

        if (ax is None):
            ax = plt.gca()

        ax.errorbar(
            self.flux["freq"],
            self.flux["values"][:,stokes_index],
            xerr=self.flux["bandwidth"] / 2,
            yerr=self.flux["errors"][:,stokes_index],
            ls='none',
            marker=marker, c=c,
        )

        if label_axes:
            ax.set_xlabel(r"$\nu$ (Hz)")
            ax.set_ylabel(r"$S_\nu$ (Jy)")
