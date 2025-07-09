"""Provide the stokesMap class for handling astronomical maps.

Use the Stokes Parameter convention to load, manage, and manipulate map data
with various projections and polarization conventions. Compatible with other
package functions.
"""

import logging

import astropy.constants as astropy_const
import astropy.convolution as astropy_conv
import astropy.units as astropy_u
import healpy as hp
import numpy as np
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix

from mapext.core.projection import healpix_coordsys_to_frame, reproject
from mapext.core.stokes import get_stokes_value_mapping, queryable_parameters
from mapext.core.utils import string_to_astropy_unit

logger = logging.getLogger(__name__)

__all__ = ["stokesMap"]


class stokesMap:
    """Class to hold astronomical maps using Stokes Parameter convention."""

    def __init__(self, *args, assume_v_0=True, **kwargs):
        """Initialise stokesMap object.

        Parameters
        ----------
        assume_v_0 : bool, optional
            Assume Stokes' V parameter equal to zero, by default True
        *args : tuple
            Positional arguments for additional initialization options.
        **kwargs : dict
            Keyword arguments for specifying maps or other initialization options.
        """
        logger.debug("Initializing stokesMap with assume_v_0=%s", assume_v_0)
        self.assume_v_0 = assume_v_0
        self._maps_cached = []
        self._calibration = {}

        # Initialise set parameters
        self._pol_convention = None
        self._unit = None
        self._frequency = None
        self._wavelength = None
        self._projection = None
        self._resolution = None
        self._shape = None

        # CASE 1: Keyword arguments provided to specify maps
        if kwargs:
            logging.info("Loading map data from keyword arguments.")
            self.load_from_kwargs(**kwargs)

        # CASE 2: Loadfile provided to specify maps
        elif (len(args) == 1) and isinstance(args[0], str):
            logging.info("Loading map data from file: %s", args[0])
            self.load_from_file(args[0])

    # ==========================================================================
    # loading functions
    def load_from_kwargs(self, **kwargs):
        """Load map data where data is directly supplied.

        Parameters
        ----------
        I : numpy array, optional
            Stokes I map data
        Q : numpy array, optional
            Stokes Q map data
        U : numpy array, optional
            Stokes U map data
        V : numpy array, optional
            Stokes V map data
        P : numpy array, optional
            Total polarisation data
        A : numpy array, optional
            Polarisation angle data
        PF : numpy array, optional
            Polarisation fraction data
        kwargs : dict
            Additional keyword arguments for specifying maps or other options.
        """
        logger.debug("Loading data via kwargs: %s", list(kwargs.keys()))
        for param in kwargs.keys():
            if (param.upper() in queryable_parameters) and (kwargs[param] is not None):
                self._maps_cached.append(param.upper())
                setattr(self, f"_{param.upper()}", kwargs[param])
            else:
                logger.error("Unrecognized parameter: %s", param)
                raise ValueError(
                    f"Parameter {param} not recognised or not supported by stokesMap."
                )

    def load_from_file(self, filename):
        """Load map data from a filename.

        Parameters
        ----------
        filename : str, path
            Path to the file containing map data

        Raises
        ------
        ValueError
            Projection type not supported yet by code
        """
        logger.debug("Reading map file: %s", filename)
        with open(filename) as f:
            load_data = yaml.safe_load(f)

        # Load name
        if "name" in load_data:
            self.name = load_data["name"]
            if "shortname" in load_data:
                self.shortname = load_data["shortname"]
            else:
                self.shortname = self.name
        # Load frequency information if supplied
        if "frequency" in load_data:
            logger.debug("Found frequency in file.")
            if isinstance(load_data["frequency"], dict):
                self.frequency = load_data["frequency"]["center"] * astropy_u.Hz
            elif isinstance(load_data["frequency"], float):
                self.frequency = load_data["frequency"] * astropy_u.Hz
            logger.info(
                "Frequency loaded from file: %s Hz", self.frequency.to(astropy_u.Hz)
            )

        # Load wavelength information if supplied
        if "wavelength" in load_data:
            if isinstance(load_data["wavelength"], dict):
                self.wavelength = load_data["frequency"]["center"] * astropy_u.m
            elif isinstance(load_data["wavelength"], float):
                self.wavelength = load_data["wavelength"] * astropy_u.m
            logger.info(
                "Wavelength loaded from file: %s m", self.wavelength.to(astropy_u.m)
            )

        # Load beam information if supplied
        if "beamwidth" in load_data:
            self._resolution = load_data["beamwidth"]["major_amin"] * astropy_u.arcmin
        
        # Load epoch information if supplied
        if "epoch" in load_data:
            setattr(
                self,
                f"epoch",
                load_data["epoch"],
            )

        for stokes_param, stokes_dict in load_data["stokes"].items():
            # Load data array
            if "data" in stokes_dict:
                projection = stokes_dict["data"].get("projection", "WCS")

                if projection.upper() in ["WCS", "CAR"]:
                    data, proj, shape = self.load_wcs_map(stokes_dict["data"])
                    if self.projection is None:
                        self.projection = proj
                    elif self.projection != proj:
                        raise ValueError(
                            f"Projection {self.projection} does not match {proj}"
                        )
                    if self.shape is None:
                        self.shape = shape
                    elif self.shape != shape:
                        raise ValueError(f"Shape {self.shape} does not match {shape}")
                    setattr(
                        self,
                        f"_{stokes_param.upper()}_MAP",
                        data,
                    )

                elif projection.upper() in ["HPX", "HEALPIX"]:
                    data, proj, shape = self.load_hpx_map(stokes_dict["data"])
                    if self.projection is None:
                        self.projection = proj
                    elif self.projection != proj:
                        logger.warning(
                            "Projection %s does not match %s. Proceeding with new projection.",
                            self.projection,
                            proj,
                        )
                    if self.shape is None:
                        self.shape = shape
                    elif self.shape != shape:
                        raise ValueError(f"Shape {self.shape} does not match {shape}")
                    setattr(self, f"_{stokes_param.upper()}_MAP", data)

                else:
                    raise ValueError(f"Projection {projection} not supported")

                if "nullval" in stokes_dict["data"]:
                    d = getattr(
                        self,
                        f"_{stokes_param.upper()}_MAP",
                        hp.read_map(stokes_dict["data"]["filename"]),
                    )
                    d[d == stokes_dict["data"]["nullval"]] = np.nan
                    setattr(self, f"_{stokes_param.upper()}_MAP", d)
                    logger.info(
                        "Loaded %s map with null value %s set to nan",
                        stokes_param.upper(),
                        stokes_dict["data"]["nullval"],
                    )
                self._maps_cached.append(stokes_param.upper())

            # Load uncertainty array if supplied
            if "uncertainty" in stokes_dict:
                logging.info(
                    "Loading uncertainty data for %s from %s",
                    stokes_param.upper(),
                    stokes_dict["uncertainty"]["filename"],
                )
                projection = stokes_dict["data"].get("projection", "WCS")
                if projection.upper() in ["WCS", "CAR"]:
                    setattr(
                        self,
                        f"_{stokes_param.upper()}_UNC",
                        fits.open(stokes_dict["uncertainty"]["filename"])[0].data,
                    )
                elif projection.upper() in ["HPX", "HEALPIX"]:
                    setattr(
                        self,
                        f"_{stokes_param.upper()}_UNC",
                        hp.read_map(stokes_dict["uncertainty"]["filename"]),
                    )
                else:
                    logger.warning(
                        "Uncertainty data provided for %s but projection %s not supported. Proceding without uncertainty maps.",
                        stokes_param.upper(),
                        projection,
                    )
            # Load calibration information if supplied
            if "calibration" in stokes_dict:
                self._calibration[stokes_param.upper()] = stokes_dict["calibration"]

    # ==========================================================================
    # map loading functions
    # Load data and projection from a file
    def load_wcs_map(self, settings):
        """Load WCS map data and projection.

        Parameters
        ----------
        settings : dict
            Dictionary containing the settings for loading the WCS map, including the filename and header index.

        Returns
        -------
        tuple
            A tuple containing the data array and the WCS projection object.
        """
        # Load data
        data = fits.open(settings["filename"])[settings.get("HDR", 0)].data
        # Load projection
        hdr = fits.open(settings["filename"])[settings.get("HDR", 0)].header
        proj = WCS(hdr)
        shape = data.shape
        # Check units against _unit
        if settings.get("units", None) is not None:
            if self._unit is None:
                self._unit = string_to_astropy_unit(settings["units"])
                if self._unit.is_equivalent(astropy_u.K):
                    self._unit_temp = 'cmb' if settings.get("temp_cmb", False) else 'rj'
            elif self._unit != string_to_astropy_unit(settings["units"]):
                raise ValueError(
                    f"Unit {self._unit} does not match {settings['units']}"
                )
        return data, proj, shape

    def load_hpx_map(self, settings):
        """Load HEALPix map data and projection.

        Parameters
        ----------
        settings : dict
            Dictionary containing the settings for loading the HEALPix map, including the filename and optional transformation.

        Returns
        -------
        tuple
            A tuple containing the data array and the HEALPix projection object.
        """
        # Load data
        data = hp.read_map(settings["filename"], field=settings.get("tform", None))
        data = np.array(data, dtype=float)
        data = data.flatten()
        # Load projection
        hdr = fits.open(settings["filename"])[settings.get("HDR", 1)].header
        try:
            frame = healpix_coordsys_to_frame[hdr["COORDSYS"]]
        except KeyError:
            logger.warning("COORDSYS keyword not found or recognised. Assuming HEALPix frame is galactic")
            frame = 'galactic'
        proj = HEALPix(
            nside=hdr["NSIDE"],
            order="RING",
            frame=frame,
        )
        shape = np.array([HEALPix.npix])
        # Check units against _unit
        if settings.get("units", None) is not None:
            if self._unit is None:
                self._unit = string_to_astropy_unit(settings["units"])
                if self._unit.is_equivalent(astropy_u.K):
                    self._unit_temp = 'cmb' if settings.get("temp_cmb", False) else 'rj'
            elif self._unit != string_to_astropy_unit(settings["units"]):
                raise ValueError(
                    f"Unit {self._unit} does not match {settings['units']}"
                )
        return data, proj, shape

    # ==========================================================================
    # pol_convention propery
    # setting polarisation convention used in this stokesMap object and managing
    # conversion
    @property
    def pol_convention(self):
        """Polarization convention used in this dataset.

        Returns
        -------
        str
            `COSMO` or `IAU` depending on the convention adopted
        """
        return self._pol_convention

    @pol_convention.setter
    def pol_convention(self, new_pol_convention):
        """Setter for Polarization convention.

        Parameters
        ----------
        new_pol_convention : str
            polarisation convention represented by data, to be adopted by the stokesMap object

        Raises
        ------
        ValueError
            pol_convention must be wither COMSO or IAU
        """
        if new_pol_convention.upper() in ["COSMO", "IAU"]:
            self._pol_convention = new_pol_convention.upper()
        else:
            raise ValueError("pol_convention must be either COSMO or IAU")

    def switch_pol_conventions(self):
        """Switch between IAU and COSMO polarization conventions.

        Switch is performed by reversing the sign on the Stokes U and/or polarization angle (A) datasets stored in the object.

        Raises
        ------
        ValueError
            Polarisation convention currently stored is not recognised
        """
        logger.debug(
            "Switching polarization conventions from: %s", self._pol_convention
        )
        if hasattr(self, "_U"):
            logger.debug("Flipping sign of Stokes U")
            self._U = -1 * self._U
        if hasattr(self, "_A"):
            logger.debug("Flipping sign of polarization angle A")
            self._A = -1 * self._A
        else:
            logger.warning("No Stokes U or polarization angle data to switch")

        if self._pol_convention == "COSMO":
            self._pol_convention = "IAU"
        elif self._pol_convention == "IAU":
            self._pol_convention = "COSMO"
        else:
            raise ValueError(f"pol_convention ({self._pol_convention}) not recognised")

    # ==========================================================================
    # projection tools
    @property
    def projection(self):
        """Projection object used in the map (WCS or HEALPix).

        Returns
        -------
        WCS or HEALPix
            The map projection associated with the data.
        """
        return self._projection

    @projection.setter
    def projection(self, proj):
        """Set the projection object for the map.

        Can accept either a WCS object or HEALPix object.

        Parameters
        ----------
        proj : WCS or HEALPix
            The projection object associated with the map.
        """
        if isinstance(proj, (HEALPix, WCS)):
            self._projection = proj
        else:
            raise ValueError("Projection must be either a WCS or HEALPix object.")

    def unit_classification(self):
        """
        Classify a unit as 'temperature', 'flux', 'flux density', or 'unknown'.

        Parameters
        ----------
        unit : astropy.units.Unit or astropy.units.Quantity

        Returns
        -------
        str : classification result
        """
        if isinstance(self._unit, astropy_u.Quantity):
            unit = self._unit.unit
        else:
            unit = self._unit

        if unit.is_equivalent(astropy_u.K):
            return "temperature"
        elif unit.is_equivalent(astropy_u.Jy):
            return "flux density"
        elif unit.is_equivalent(astropy_u.W / astropy_u.m**2) or unit.is_equivalent(astropy_u.erg / (astropy_u.s * astropy_u.cm**2)):
            return "flux"
        else:
            return "unknown"

    def set_projection(self, proj, map_shape=None, **kwargs):
        """Set the projection for the map, either as a HEALPix or WCS and transform all maps.

        Parameters
        ----------
        proj : HEALPix or None
            If a HEALPix object is provided, it will be directly used for the projection.
        wcs_header : fits.Header or None
            If using WCS, the header containing WCS information.
        map_shape : tuple or None
            The shape of the map, needed for WCS if provided.
        kwargs : dict
            Additional keyword arguments for WCS or HEALPix configuration.
        """
        logger.debug("Setting new projection: %s", type(proj).__name__)
        # Validate the input
        if isinstance(proj, HEALPix):
            new_projection = proj
            new_shape = proj.npix
        elif isinstance(proj, WCS):
            new_projection = proj
            if np.array(map_shape).shape[0] != proj.naxis:
                raise ValueError(
                    "Map shape does not match the number of axes in the WCS projection."
                )
            new_shape = map_shape
        else:
            raise ValueError("Projection object must be of type HEALPix or WCS.")

        for stokes in self._maps_cached:
            stokes_map = getattr(self, f"_{stokes}_MAP")
            if stokes_map is not None:
                logger.debug("Reprojecting %s map to new projection", stokes)
                if isinstance(new_projection, HEALPix):
                    stokes_map = reproject(stokes_map, self.projection, new_projection, preserve_flux=True if self.unit_classification() == "flux density" else False)
                elif isinstance(new_projection, WCS):
                    stokes_map = reproject(
                        stokes_map,
                        self.projection,
                        new_projection,
                        shape_out=new_shape,
                    )
                setattr(self, f"_{stokes}_MAP", stokes_map)

        self.projection = new_projection
        self.shape = map_shape

    @property
    def shape(self):
        """Shape of the map data array.

        Returns
        -------
        tuple
            Shape of the map data.
        """
        return self._shape

    @shape.setter
    def shape(self, shp):
        """Set the shape of the map data array.

        Parameters
        ----------
        shp : tuple
            Shape to be associated with the map data.
        """
        self._shape = np.array(shp)

    # ==========================================================================
    # frequency and wavelength froperties
    @property
    def frequency(self):
        """Frequency of Stokes data.

        Returns
        -------
        astropy Quantity
            Astropy quantity with units of Hz (Hertz) representing the frequency of map data
        """
        if self._frequency is not None:
            return self._frequency
        if self._wavelength is not None:
            return (astropy_const.c / self._wavelength).to(astropy_u.Hz)
        return ValueError("No frequency or wavelength has been provided")

    @property
    def wavelength(self):
        """Wavelength of Stokes data.

        Returns
        -------
        astropy Quantity
            Astropy quantity with units of m (meters) representing the frequency of map data
        """
        if self._wavelength is not None:
            return self._wavelength
        if self._frequency is not None:
            return (astropy_const.c / self._frequency).to(astropy_u.m)
        return ValueError("No frequency or wavelength has been provided")

    @frequency.setter
    def frequency(self, new_frequency):
        """Setter for map data frequency.

        Parameters
        ----------
        new_frequency : astropy Quantity, float or int
            frequency of map data, overwrites previous values of frequency and wavelength. If value is given as a float or integer, the unit will be interpreted as GHz if the value is <1e6, and in Hz if >=1e6.

        Raises
        ------
        ValueError
            Please give frequency as either an astropy quantity, or as a float (see docs for details of interpretation)
        """
        logger.debug("Setting frequency to: %s", new_frequency)
        self._wavelength = None
        if new_frequency is None:
            self._frequency = None
        if isinstance(new_frequency, astropy_u.Quantity):
            self._frequency = new_frequency.to(astropy_u.Hz)
        elif isinstance(new_frequency, float) or isinstance(new_frequency, int):
            if new_frequency < 1e6:
                logger.warning("Assuming frequency has been given in GHz")
                self._frequency = new_frequency * 1e9 * astropy_u.Hz
            elif new_frequency >= 1e6:
                logger.warning("Assuming frequency has been given in Hz")
                self._frequency = new_frequency * astropy_u.Hz
        else:
            raise ValueError(
                "Please give frequency as either an astropy quantity, or as a float (see docs for details of interpretation)"
            )

    @wavelength.setter
    def wavelength(self, new_wavelength):
        """Setter for map data wavelength.

        Parameters
        ----------
        new_wavelength : astropy Quantity, float or int
            wavelength of map data, overwrites previous values of frequency and wavelength. If value is given as a float or integer, the unit will be interpreted as um if the value is >1e-2, and in m if <=1e-2.

        Raises
        ------
        ValueError
            Please give wavelength as either an astropy quantity, or as a float (see docs for details of interpretation)
        """
        logger.debug("Setting wavelength to: %s", new_wavelength)
        self._frequency = None
        if new_wavelength is None:
            self._wavelength = None
        if isinstance(new_wavelength, astropy_u.Quantity):
            self._wavelength = new_wavelength.to(astropy_u.m)
        elif isinstance(new_wavelength, float) or isinstance(new_wavelength, int):
            if new_wavelength > 1e-2:
                logger.warning("Assuming wavelength has been given in m")
                self._wavelength = new_wavelength * astropy_u.m
            elif new_wavelength <= 1e-2:
                logger.warning("Assuming frequency has been given in um")
                self._wavelength = new_wavelength * 1e-6 * astropy_u.m
        else:
            raise ValueError(
                "Please give wavelength as either an astropy quantity, or as a float (see docs for details of interpretation)"
            )

    # ==========================================================================
    # stokes component properties
    def _get_stokes_map(self, stokes_type):
        """Helper to generate a Stokes map of the specified type."""
        logger.debug("Accessing Stokes map for type: %s", stokes_type)
        func = get_stokes_value_mapping(
            stokes_type,
            self._maps_cached,
            assume_v_0=getattr(self, "assume_v_0", None),
        )
        return func(
            **{param: getattr(self, f"_{param}_MAP") for param in self._maps_cached}
        )

    @property
    def I(self):  # noqa: E743
        """Stokes I map."""
        return self._get_stokes_map("I")

    @property
    def Q(self):
        """Stokes Q map."""
        return self._get_stokes_map("Q")

    @property
    def U(self):
        """Stokes U map."""
        return self._get_stokes_map("U")

    @property
    def V(self):
        """Stokes V map."""
        return self._get_stokes_map("V")

    @property
    def P(self):
        """Total polarized intensity (P) map."""
        return self._get_stokes_map("P")

    @property
    def A(self):
        """Polarisation angle map."""
        return self._get_stokes_map("A")

    @property
    def PF(self):
        """Polarisation fraction (PF) map."""
        return self._get_stokes_map("PF")

    # ==========================================================================
    # Map manipulation functions
    def smooth_to(self, fwhm=None, preserve='flux'):
        """Smooth the map to a specified FWHM.

        Parameters
        ----------
        fwhm : float or astropy Quantity, optional
            Full width at half maximum (FWHM) in degrees or arcminutes. If not provided, defaults to 1 arcminute.
        """
        if fwhm is None:
            fwhm = 1 * astropy_u.degree
        elif isinstance(fwhm, (float, int)):
            fwhm = fwhm * astropy_u.deg
            logger.info("Assuming FWHM is given in degrees: %s", fwhm)

        # determine smoothing factor
        smooth_gaussian_deg = (
            np.sqrt(fwhm**2 - self._resolution**2).to(astropy_u.deg).value
        )

        logger.debug("Smoothing maps to FWHM: %s", fwhm)
        for stokes in self._maps_cached:
            stokes_map = getattr(self, f"_{stokes}_MAP")
            if (stokes_map is not None) and isinstance(self.projection, HEALPix):
                logger.debug("Smoothing %s map", stokes)
                mask = np.isfinite(stokes_map) & (stokes_map != hp.UNSEEN) & (stokes_map != 0.0) & (stokes_map > -1e30)
                stokes_map[~mask] = 0.0
                stokes_map_clean = np.copy(stokes_map)
                stokes_map_clean[~mask] = 0.0
                smoothed_map = np.array(
                    hp.smoothing(stokes_map_clean, fwhm=np.radians(smooth_gaussian_deg)),
                    dtype=float,
                )
                smoothed_map[~mask] = np.nan
                setattr(self, f"_{stokes}_MAP", smoothed_map)

            elif (stokes_map is not None) and isinstance(self.projection, WCS):
                logger.debug("Smoothing %s map", stokes)
                stokes_map[np.isfinite(stokes_map) == False] = np.nan  # noqa: E712
                kernel = astropy_conv.Gaussian2DKernel(
                    x_stddev=smooth_gaussian_deg / self.projection.wcs.cdelt[0],
                    y_stddev=smooth_gaussian_deg / self.projection.wcs.cdelt[1],
                )
                smoothed_map = astropy_conv.convolve(
                    stokes_map, kernel, normalize_kernel=True, boundary="fill"
                )
                setattr(self, f"_{stokes}_MAP", smoothed_map)
        self._resolution = fwhm

    def set_units(self, new_units):
        """Convert the map units to a new set of units.

        Parameters
        ----------
        new_units : astropy.units.Unit
            The new units to convert the map to.
        """
        if type(new_units) is str:
            new_units = string_to_astropy_unit(new_units)
        conv_start = 1 * self._unit
        if new_units != self._unit:
            # first convert to Jy.sr
            if self._unit in [astropy_u.W/(astropy_u.m**2 * astropy_u.sr)]:
                conv_int = (conv_start).to(astropy_u.Jy/astropy_u.sr, equivalencies=astropy_u.spectral_density(self.frequency))
            elif (self._unit in [astropy_u.K, astropy_u.mK, astropy_u.uK, astropy_u.nK]) and (self._unit_temp == 'rj'):
                # print('RJ')
                conv_int = (conv_start).to(astropy_u.Jy/astropy_u.sr, equivalencies=astropy_u.brightness_temperature(self.frequency))
            elif (self._unit in [astropy_u.K, astropy_u.mK, astropy_u.uK, astropy_u.nK]) and (self._unit_temp == 'cmb'):
                # print('CMB')
                from astropy.cosmology import Planck15
                conv_int = (conv_start).to(
                    astropy_u.Jy/astropy_u.sr, equivalencies=astropy_u.thermodynamic_temperature(self.frequency, Planck15.Tcmb0))
            elif (self._unit in [astropy_u.Jy/astropy_u.sr, string_to_astropy_unit('MJy/sr')]):
                conv_int = (conv_start).to(astropy_u.Jy/astropy_u.sr)
            elif self._unit in [astropy_u.Jy/astropy_u.beam, astropy_u.mJy/astropy_u.beam]:
                conv_int = (conv_start).to(astropy_u.Jy/astropy_u.sr, equivalencies=astropy_u.beam_angular_area(self.get_beam_area()))
            elif self._unit in [astropy_u.Jy/(astropy_u.pixel**2)]:
                pixel = self.pixelsize
                conv_int = ((conv_start) / pixel).to(astropy_u.Jy/astropy_u.sr)
            else:
                raise ValueError(f'Unit {self._unit} not recognised as flux density units')
            # convert to final units
            # first convert to Jy.sr
            if new_units in [astropy_u.W/(astropy_u.m**2 * astropy_u.sr)]:
                conv_final = conv_int.to(new_units, equivalencies=astropy_u.spectral_density(self.frequency))
            elif new_units in [astropy_u.K]:
                conv_final = conv_int.to(new_units, equivalencies=astropy_u.brightness_temperature(self.frequency))
            elif new_units in [astropy_u.Jy/astropy_u.sr]:
                conv_final = conv_int
            elif new_units in [astropy_u.Jy/astropy_u.beam]:
                conv_final = conv_int.to(new_units, equivalencies=astropy_u.beam_angular_area(self.get_beam_area()))
            elif new_units in [astropy_u.Jy/(astropy_u.pixel**2)]:
                pixel = self.pixelsize
                conv_final = (conv_int * pixel).to(astropy_u.Jy/(astropy_u.pixel**2))
            else:
                raise ValueError(f'Unit {new_units} not recognised as flux density units')
            
            for stokes in self._maps_cached:
                stokes_map = getattr(self, f"_{stokes}_MAP")
                stokes_map[np.isfinite(stokes_map) == False] = np.nan # noqa: E712
                if stokes_map is not None:
                    setattr(self, f"_{stokes}_MAP", (stokes_map / conv_start * conv_final).value)
            self._unit = new_units
        return
    
    @property
    def pixelsize(self):
        """Get the pixel size of the map in square degrees."""
        if isinstance(self.projection, WCS):
            return abs(self.projection.wcs.cdelt[0]*self.projection.wcs.cdelt[1]) * (astropy_u.degree/astropy_u.pixel)**2
        raise ValueError("Cannot determine pixel size for non-WCS projections.")