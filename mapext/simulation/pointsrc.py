"""Provide functionality for generating point sources in simulated maps.

Simulate point sources using HEALPix and WCS projections.
"""

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy_healpix import HEALPix

from mapext.simulation.core import stokesMapSimulationComponent

__all__ = [
    "pointSource",
]


class pointSource(stokesMapSimulationComponent):
    """Simulate point sources at specified coordinates in a map.

    This class generates point sources and places them at specific coordinates in
    either HEALPix or WCS projections.

    Attributes
    ----------
    projection : HEALPix or WCS
        The projection system (either HEALPix or WCS).
    shape_out : tuple
        The shape of the output map (used for WCS).
    """

    def run_simulation(
        self,
        source_coords=SkyCoord(0, 0, unit="degree", frame="galactic"),
        I=None,
        Q=None,
        U=None,
        V=None,
        P=None,
        A=None,
        PF=None,
        fwhm_deg=None,
    ):
        """Generate a point source map and place the source at the specified coordinates.

        Parameters
        ----------
        source_coords : tuple, optional
            Coordinates for the point source (either in pixel coordinates for WCS or
            in angular coordinates (theta, phi) for HEALPix).
        I : float, optional
            Peak intensity for the I component.
        Q : float, optional
            Peak intensity for the Q component.
        U : float, optional
            Peak intensity for the U component.
        V : float, optional
            Peak intensity for the V component.
        P : float, optional
            Peak intensity for the P component.
        A : float, optional
            Peak intensity for the A component.
        PF : float, optional
            Peak intensity for the PF component.
        fwhm_deg : float, optional
            Full width at half maximum (FWHM) for the point source.

        Returns
        -------
        dict
            A dictionary containing the generated map with the point source.
        """
        peak_intensity = {
            name: value
            for name, value in {
                "I": I,
                "Q": Q,
                "U": U,
                "V": V,
                "P": P,
                "A": A,
                "PF": PF,
            }.items()
            if value is not None
        }
        if self.assume_v_0:
            peak_intensity["V"] = 0.0

        # Generate an unconvolved point source map
        if isinstance(self._projection, HEALPix):
            ptsrc = self._generate_healpix_point_source(source_coords)

        elif isinstance(self._projection, WCS):
            ptsrc = self._generate_wcs_point_source(source_coords)

        else:
            raise ValueError("Projection must be either HEALPix or WCS")

        # Convolve the point source map with a Gaussian kernel if fwhm_deg is provided
        if fwhm_deg is not None:
            if isinstance(self._projection, HEALPix):
                ptsrc = hp.smoothing(ptsrc, fwhm=fwhm_deg * np.pi / 180.0)
            elif isinstance(self._projection, WCS):
                from astropy.convolution import Gaussian2DKernel, convolve

                kernel = Gaussian2DKernel(
                    x_stddev=fwhm_deg / (2.355 * self._projection.wcs.cdelt[0])
                )
                ptsrc = convolve(ptsrc, kernel, boundary="wrap")

        outputs = {}

        for comp, val in peak_intensity.items():
            if comp == "A":
                outputs["A"] = np.ones(self._shape_out) * val
            else:
                outputs[comp] = ptsrc * (val / np.nanmax(ptsrc))

        return outputs

    def _generate_healpix_point_source(self, source_coords):
        """Generate a HEALPix map with a point source at the specified coordinates.

        The source coordinates can be provided as a SkyCoord object, tuple (theta, phi), or pixel index.

        Parameters
        ----------
        source_coords : SkyCoord, tuple, or int
            - If SkyCoord: Angular coordinates (theta, phi) of the point source.
            - If tuple: A tuple (theta, phi) in radians.
            - If int: Pixel index for the point source.

        Returns
        -------
        numpy.ndarray
            A HEALPix map with a point source.
        """
        # Initialize a blank HEALPix map
        nside = self._projection.nside
        map_data = np.zeros(hp.nside2npix(nside))

        # Initialize the pixel index
        pix = None

        # If the source coordinates are given as a SkyCoord object
        if isinstance(source_coords, SkyCoord):
            # Convert to theta and phi in radians
            theta, phi = (
                source_coords.spherical.lat.radian,
                source_coords.spherical.lon.radian,
            )
            # Convert angular coordinates (theta, phi) to pixel index
            pix = int(hp.ang2pix(nside, theta, phi))

        # If source_coords is a tuple (theta, phi)
        elif isinstance(source_coords, tuple) and len(source_coords) == 2:
            # Otherwise, assume source_coords is a tuple (theta, phi)
            theta, phi = source_coords
            # Validate angular coordinates
            if not (0 <= theta <= np.pi):
                raise ValueError(
                    f"theta out of bounds: {theta} (must be between 0 and pi)"
                )
            if not (0 <= phi < 2 * np.pi):
                raise ValueError(
                    f"phi out of bounds: {phi} (must be between 0 and 2*pi)"
                )
            # Convert angular coordinates (theta, phi) to pixel index
            pix = int(hp.ang2pix(nside, theta, phi))

        # If source_coords is an integer (pixel index), use it directly
        elif isinstance(source_coords, int):
            pix = source_coords

        else:
            raise ValueError(
                "Invalid source_coords format. Must be SkyCoord, tuple (theta, phi), or pixel index."
            )

        # Validate pixel index
        if pix is None:
            raise ValueError(
                "Pixel index is not defined. Something went wrong with the coordinates."
            )

        # Set the intensity at the specified pixel
        map_data[pix] += 1

        return map_data

    def _generate_wcs_point_source(self, source_coords):
        """Generate a WCS map with a point source at the specified coordinates.

        The source coordinates can be provided as a SkyCoord object, tuple (ra, dec), or pixel coordinates (x, y).

        Parameters
        ----------
        source_coords : SkyCoord, tuple, or int
            - If SkyCoord: Angular coordinates (ra, dec) of the point source.
            - If tuple: A tuple (ra, dec) in degrees (or the relevant units of the WCS).
            - If tuple of integers: Pixel coordinates (x, y).

        Returns
        -------
        numpy.ndarray
            A WCS map with a point source.
        """
        # Initialize a blank WCS map
        map_data = np.zeros(self._shape_out)

        # If the source coordinates are given as a SkyCoord object (angular)
        if isinstance(source_coords, SkyCoord):
            # Convert to pixel coordinates using the WCS
            x, y = self._projection.world_to_pixel(source_coords)
        elif isinstance(source_coords, tuple) and len(source_coords) == 2:
            # If it's a tuple (ra, dec) in degrees or pixel coordinates (x, y)
            if isinstance(source_coords[0], (int, float)) and isinstance(
                source_coords[1], (int, float)
            ):
                # If source_coords is a tuple of (x, y), it's in pixel space
                x, y = source_coords
            else:
                # If it's a tuple (ra, dec), we convert angular coordinates to pixels
                ra, dec = source_coords
                sky_coord = SkyCoord(ra=ra, dec=dec, unit="deg")
                x, y = self._projection.world_to_pixel(sky_coord)
        else:
            raise ValueError(
                "Invalid source_coords format. Must be SkyCoord or tuple (ra, dec) or (x, y)."
            )

        if isinstance(x, np.ndarray):
            x = x.item()  # Convert to scalar if it's a numpy array
        if isinstance(y, np.ndarray):
            y = y.item()  # Convert to scalar if it's a numpy array

        # Validate pixel coordinates
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(
                f"Pixel coordinates must be numerical. Got x: {type(x)}, y: {type(y)}."
            )
        if not (0 <= x < self._shape_out[1]) or not (0 <= y < self._shape_out[0]):
            raise ValueError(
                f"Pixel coordinates {(x, y)} are out of bounds. Must be between (0, 0) and {self._shape_out[1]-1}, {self._shape_out[0]-1}."
            )

        # Place the point source at the specified pixel
        map_data[int(y), int(x)] += 1

        return map_data
