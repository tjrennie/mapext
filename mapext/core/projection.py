"""Provide functionality for reprojecting data between different coordinate systems.

This includes support for WCS and HEALPix formats.
"""

import logging

import numpy as np
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from healpy import ud_grade
from reproject import reproject_from_healpix, reproject_interp, reproject_to_healpix

__all__ = [
    "reproject",
]

logger = logging.getLogger(__name__)

def reproject(data, proj1, proj2, shape_out=None, preserve_flux=False):
    """Reproject data from one projection to another.

    This function supports reprojecting between WCS and HEALPix formats.

    Parameters
    ----------
    data : array-like or astropy.units.Quantity
        The data to be reprojected. Can be a 2D array or a HEALPix map.
    proj1 : astropy.wcs.WCS or astropy_healpix.HEALPix
        The projection of the input data.
    proj2 : astropy.wcs.WCS or astropy_healpix.HEALPix
        The projection to which the data should be reprojected.
    shape_out : tuple, optional
        The shape of the output data. If not provided, it will be determined
        based on the projection.
    preserve_flux : bool, optional
        If True, reprojected data will conserve total flux (e.g., for intensity in Jy/sr).
        If False, values will be averaged without scaling (e.g., for temperature in K).
        This only applies to HEALPix-to-HEALPix regridding.

    Returns
    -------
    mapout : array-like or Quantity
        The reprojected data.
    """
    if isinstance(data, Quantity):
        unit = data.unit
        data = data.value
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
            logger.error("Cannot reproject between different frames.")
            raise ValueError("Cannot reproject between different frames.")

        power = -2 if preserve_flux else 0
        mapout = ud_grade(
            data,
            nside_out=proj2.nside,
            order_in="NESTED" if proj1.order == "nested" else "RING",
            order_out="NESTED" if proj2.order == "nested" else "RING",
            power=power,
        )

    else:
        logger.error(f"Unsupported projection mapping: {type(proj1)} -> {type(proj2)}")
        raise ValueError(
            f"Unsupported projection mapping: {type(proj1)} -> {type(proj2)}"
        )

    return mapout * unit
