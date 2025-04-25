"""Provide functionality for reprojecting data between different coordinate systems.

This includes support for WCS and HEALPix formats.
"""

import numpy as np
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from healpy import ud_grade
from reproject import reproject_from_healpix, reproject_interp, reproject_to_healpix

__all__ = [
    "reproject",
]


def reproject(data, proj1, proj2, shape_out=None):
    """Reproject data from one projection to another.

    This function supports reprojecting between WCS and HEALPix formats.

    Parameters
    ----------
    data : array-like
        The data to be reprojected. Can be a 2D array or a HEALPix map.
    proj1 : WCS or HEALPix
        The projection of the input data.
    proj2 : WCS or HEALPix
        The projection to which the data should be reprojected.
    shape_out : tuple, optional
        The shape of the output data. If not provided, it will be determined
        based on the projection.

    Returns
    -------
    mapout : array-like
        The reprojected data.
    """
    if isinstance(data, Quantity):
        unit = data.unit
    else:
        unit = 1

    # WCS -> WCS
    if isinstance(proj1, WCS) and isinstance(proj2, WCS):
        mapout, mask = reproject_interp(
            (data, proj1),
            proj2,
            shape_out=shape_out,
            return_footprint=True,
        )
        mapout[mask == 0] = np.nan

    # WCS -> HEALPix
    elif isinstance(proj1, WCS) and isinstance(proj2, HEALPix):
        mapout, mask = reproject_to_healpix(
            (data, proj1),
            nside=proj2.nside,
            projection_type="ring" if proj2.order == "ring" else "nested",
            frame=proj2.frame.name,
            return_footprint=True,
        )
        mapout[mask == 0] = np.nan

    # HEALPix -> WCS
    elif isinstance(proj1, HEALPix) and isinstance(proj2, WCS):
        mapout, mask = reproject_from_healpix(
            (data, proj1),
            proj2,
            shape_out=shape_out,
            return_footprint=True,
        )
        mapout[mask == 0] = np.nan

    # HEALPix -> HEALPix
    elif isinstance(proj1, HEALPix) and isinstance(proj2, HEALPix):
        if proj1.frame.name != proj2.frame.name:
            raise ValueError("Cannot reproject between different frames.")

        mapout = ud_grade(
            data,  # Original HEALPix map
            nside_out=proj2.nside,  # Target NSIDE (from proj2)
            order_in="NESTED" if proj1.order == "nested" else "RING",
            order_out="NESTED" if proj2.order == "nested" else "RING",
            power=0,  # Use -2 for flux conservation, 0 for averaging
        )

    else:
        raise ValueError(
            f"Unsupported projection mapping: {type(proj1)} -> {type(proj2)}"
        )

    return mapout * unit
