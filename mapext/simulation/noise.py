"""Provide functionality for generating colored noise maps.

Simulate white noise using HEALPix and WCS projections.
"""

import logging

import healpy as hp
import numpy as np
from astropy.wcs import WCS
from astropy_healpix import HEALPix

from mapext.simulation.core import stokesMapSimulationComponent

logger = logging.getLogger(__name__)

__all__ = [
    "coloredNoise",
    "whiteNoise",
]


def generate_hpx_colored_noise(col=-1, rms=1, nside=32):
    """Generate a HEALPix map with colored noise.

    Parameters
    ----------
    col : int, optional
        Spectral index of the noise (default is -1).
    rms : float, optional
        Root mean square of the noise (default is 1).
    nside : int, optional
        HEALPix nside parameter (default is 32).

    Returns
    -------
    numpy.ndarray
        HEALPix map with colored noise.
    """
    logger.debug(
        f"Generating HEALPix colored noise (col={col}, rms={rms}, nside={nside})"
    )

    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=np.float64)

    cl = np.zeros_like(ell)
    cl[1:] = ell[1:] ** col

    alm = hp.synalm(cl, lmax=lmax, new=True)
    map_1f = hp.alm2map(alm, nside=nside, lmax=lmax, verbose=False)

    logger.debug("HEALPix noise map generated and normalized")

    return map_1f * (rms / np.std(map_1f))


def generate_wcs_colored_noise(col=-1, rms=1, shape=(101, 101)):
    """Generate a WCS map with colored noise.

    Parameters
    ----------
    col : float, optional
        Spectral index of the noise (default is -1).
    rms : float, optional
        Root mean square of the noise (default is 1).
    shape : list of int, optional
        Shape of the output map as [rows, cols] (default is [101, 101]).

    Returns
    -------
    numpy.ndarray
        WCS map with colored noise.
    """
    logger.debug(f"Generating WCS colored noise (col={col}, rms={rms}, shape={shape})")

    rows, cols = shape
    fx = np.fft.fftfreq(cols).reshape(1, cols)
    fy = np.fft.fftfreq(rows).reshape(rows, 1)
    f = np.sqrt(fx**2 + fy**2)

    amplitude = f**col
    amplitude[f == 0] = 0.0

    phase = np.random.uniform(0, 2 * np.pi, size=shape)
    spectrum = amplitude * (np.cos(phase) + 1j * np.sin(phase))

    map_1f = np.fft.ifft2(spectrum).real

    logger.debug("WCS noise map generated and normalized")

    return map_1f * (rms / np.std(map_1f))


class whiteNoise(stokesMapSimulationComponent):
    """Simulate white noise maps for different Stokes parameters.

    This class generates white noise maps using either HEALPix or WCS projections,
    based on the specified root mean square (RMS) values for various Stokes components.

    Methods
    -------
    run_simulation(...)
        Generate white noise maps for the specified Stokes parameters.
    """

    def run_simulation(
        self,
        I_rms=None,
        Q_rms=None,
        U_rms=None,
        V_rms=None,
        P_rms=None,
        A_rms=None,
        PF_rms=None,
        seed=None,
        all_maps_seeded=False,
    ):
        """Generate white noise maps for specified Stokes parameters.

        Parameters
        ----------
        I_rms : float, optional
            RMS value for the I Stokes parameter.
        Q_rms : float, optional
            RMS value for the Q Stokes parameter.
        U_rms : float, optional
            RMS value for the U Stokes parameter.
        V_rms : float, optional
            RMS value for the V Stokes parameter.
        P_rms : float, optional
            RMS value for the P Stokes parameter.
        A_rms : float, optional
            RMS value for the A Stokes parameter.
        PF_rms : float, optional
            RMS value for the PF Stokes parameter.
        seed : int, optional
            Random seed for reproducibility.
        all_maps_seeded : bool, optional
            If True, all maps will use the same seed.

        Returns
        -------
        dict
            A dictionary containing the generated noise maps for each Stokes parameter.
        """
        logger.info("Starting white noise simulation")
        if (seed is not None) and (all_maps_seeded is False):
            np.random.seed(seed)
            logger.debug(f"Random seed set to {seed}")

        defined_rms = {
            name: value
            for name, value in {
                "I": I_rms,
                "Q": Q_rms,
                "U": U_rms,
                "V": V_rms,
                "P": P_rms,
                "A": A_rms,
                "PF": PF_rms,
            }.items()
            if value is not None
        }
        if self.assume_v_0:
            defined_rms["V"] = 0.0
            logger.debug("Assuming V=0, V RMS set to 0")

        outputs = {}

        if isinstance(self._projection, HEALPix):
            logger.debug("Using HEALPix projection")
            for comp, rms in defined_rms.items():
                if (seed is not None) and (all_maps_seeded is True):
                    np.random.seed(seed)
                outputs[comp] = generate_hpx_colored_noise(
                    col=0,
                    rms=rms,
                    nside=self._projection.nside,
                )

        elif isinstance(self._projection, WCS):
            logger.debug("Using WCS projection")
            for comp, rms in defined_rms.items():
                if (seed is not None) and (all_maps_seeded is True):
                    np.random.seed(seed)
                outputs[comp] = generate_wcs_colored_noise(
                    col=0,
                    rms=rms,
                    shape=self._shape_out,
                )

        else:
            logger.error("Invalid projection type")
            raise ValueError("Projection must be either HEALPix or WCS")

        logger.info("White noise simulation completed")
        return outputs


class coloredNoise(stokesMapSimulationComponent):
    """Simulate colored noise maps for different Stokes parameters.

    This class generates colored noise maps using either HEALPix or WCS projections,
    based on the specified root mean square (RMS) values for various Stokes components.

    Methods
    -------
    run_simulation(...)
        Generate colored noise maps for the specified Stokes parameters.
    """

    def run_simulation(
        self,
        alpha=0,
        I_rms=None,
        Q_rms=None,
        U_rms=None,
        V_rms=None,
        P_rms=None,
        A_rms=None,
        PF_rms=None,
        seed=None,
        all_maps_seeded=False,
    ):
        """Generate colored noise maps for specified Stokes parameters.

        Parameters
        ----------
        alpha : float, optional
            Spectral index of the noise (default is 0).
        I_rms : float, optional
            RMS value for the I Stokes parameter.
        Q_rms : float, optional
            RMS value for the Q Stokes parameter.
        U_rms : float, optional
            RMS value for the U Stokes parameter.
        V_rms : float, optional
            RMS value for the V Stokes parameter.
        P_rms : float, optional
            RMS value for the P Stokes parameter.
        A_rms : float, optional
            RMS value for the A Stokes parameter.
        PF_rms : float, optional
            RMS value for the PF Stokes parameter.
        seed : int, optional
            Random seed for reproducibility.
        all_maps_seeded : bool, optional
            If True, all maps will use the same seed.

        Returns
        -------
        dict
            A dictionary containing the generated noise maps for each Stokes parameter.
        """
        logger.info("Starting colored noise simulation")

        if (seed is not None) and (all_maps_seeded is False):
            np.random.seed(seed)
            logger.debug(f"Random seed set to {seed}")

        defined_rms = {
            name: value
            for name, value in {
                "I": I_rms,
                "Q": Q_rms,
                "U": U_rms,
                "V": V_rms,
                "P": P_rms,
                "A": A_rms,
                "PF": PF_rms,
            }.items()
            if value is not None
        }
        if self.assume_v_0:
            defined_rms["V"] = 0.0
            logger.debug("Assuming V=0, V RMS set to 0")

        outputs = {}

        if isinstance(self._projection, HEALPix):
            logger.debug("Using HEALPix projection")
            for comp, rms in defined_rms.items():
                if (seed is not None) and (all_maps_seeded is True):
                    np.random.seed(seed)
                outputs[comp] = generate_hpx_colored_noise(
                    col=alpha,
                    rms=rms,
                    nside=self._projection.nside,
                )

        elif isinstance(self._projection, WCS):
            logger.debug("Using WCS projection")
            for comp, rms in defined_rms.items():
                if (seed is not None) and (all_maps_seeded is True):
                    np.random.seed(seed)
                outputs[comp] = generate_wcs_colored_noise(
                    col=alpha,
                    rms=rms,
                    shape=self._shape_out,
                )

        else:
            logger.error("Invalid projection type")
            raise ValueError("Projection must be either HEALPix or WCS")

        logger.info("Colored noise simulation completed")
        return outputs
