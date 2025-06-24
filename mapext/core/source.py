"""This module defines the astroSrc class for representing astronomical sources and their associated data."""

import csv
import json

import numpy as np
from astropy.coordinates import SkyCoord

__all__ = ['astroSrc']

class astroSrc:
    """Class to hold information pertaining to a specific astronomical source.
    
    This class is currently aimed at point sources, although will be expanded to be more flexible.
    """

    def __init__(self, name='unnamed', coords=None, frame='galactic'):
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
            raise ValueError("Coordinates must be provided as a list or a SkyCoord object.")

        if isinstance(coords, SkyCoord):
            self.coord = coords
            self.frame = coords.frame.name
        else:
            self.coord = SkyCoord(*coords, frame=frame, unit='degree')
            self.frame = frame

        self.flux = np.array([], dtype=[
            ('name', '<U40'),
            ('freq', 'float'),
            ('bandwidth', 'float'),
            ('values', 'float', (7,)),
            ('errors', 'float', (7,))
        ])

    def add_flux(self, name, freq, bandwidth, values, errors):
        """Add a flux measurement entry.

        Parameters
        ----------
        name : str
            Name or label for this flux measurement.
        freq : float
            Frequency in Hz or relevant units.
        bandwidth : float
            Bandwidth in Hz or relevant units.
        values : array-like of float
            Flux values (7 elements expected).
        errors : array-like of float
            Errors corresponding to the flux values (7 elements expected).
        """
        new_entry = np.array([(name, freq, bandwidth, values, errors)], dtype=self.flux.dtype)
        self.flux = np.append(self.flux, new_entry)

    def __repr__(self):
        return f"<astroSrc: {self.name}, Coord: {self.coord.to_string('decimal')}, Frame: {self.frame}>"

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
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                coords = [float(row['lon']), float(row['lat'])]
                frame = row.get('frame', 'galactic')

                src = cls(name=name, coords=coords, frame=frame)

                sources.append(src)
        return sources

    @classmethod
    def from_json(cls, filename):
        """Load sources from a JSON file.

        Expects a list of dicts, each with keys: name, coords (list), frame (optional).

        Returns
        -------
        list of astroSrc
        """
        with open(filename) as f:
            data = json.load(f)

        sources = []
        for item in data:
            name = item['name']
            coords = item['coords']
            frame = item.get('frame', 'galactic')

            src = cls(name=name, coords=coords, frame=frame)

            # Similar flux parsing if flux info is present
            if 'flux' in item:
                for flux_entry in item['flux']:
                    src.add_flux(
                        flux_entry['name'],
                        flux_entry['freq'],
                        flux_entry['bandwidth'],
                        flux_entry['values'],
                        flux_entry['errors']
                    )

            sources.append(src)
        return sources
