import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Beam(ABC):
    """Abstract base class for beam patterns with polarisation support."""
    
    def __init__(self, *, mueller_matrix=None):
        """Initialize the Beam with an optional Mueller matrix.

        Parameters
        ----------
        mueller_matrix : _type_, optional
            Mueller matrix defining the polarisation response of the beam. If None, defaults to the identity matrix (no polarisation effect).
        """
        self.mueller_matrix = mueller_matrix if mueller_matrix is not None else np.eye(4)
        self.kernel_radius, self.kernel_resolution = self._set_kernel_defaults()

    @property
    def mueller_matrix_normalised(self) -> bool:
        """Check if the Mueller matrix is normalised (I->I = 1).

        Returns
        -------
        bool
            True if the Mueller matrix is normalised such that M_II == 1, False otherwise.
        """
        return np.isclose(self.mueller_matrix[0,0], 1)

    def _set_kernel_defaults(self) -> tuple[float, int]:
        """Set default kernel parameters.

        Returns
        -------
        Pixel size, radius
        (float, int)
            Default pixel size and radius (in pixels) for beam kernel.
        """
        return 1.0, 100

    def _ibeam_creation_function(self, x, y):
        """Function to create the unpolarised intensity beam pattern.

        Parameters
        ----------
        x : array-like astropy Quantity
            X-coordinates in beam frame
        y : array-like astropy Quantity
            Y-coordinates in beam frame

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def _polarised_beam_creation_function(self, stokes, x, y):
        """Create polarised components based on the Mueller matrix first column.
        
        Parameters
        ----------
        stokes : str
            Stokes parameter to compute ('I', 'Q', 'U', or 'V').
        x : array-like astropy Quantity
            X-coordinates in beam frame
        y : array-like astropy Quantity
            Y-coordinates in beam frame
            
        Returns
        -------
        array-like
            Beam pattern for the specified Stokes parameter.
        """
        ibeam = self._ibeam_creation_function(x, y)
        stokes_map = {"I": 0, "Q": 1, "U": 2, "V": 3}

        if stokes not in stokes_map:
            raise ValueError("Stokes parameter must be one of 'I', 'Q', 'U', or 'V'")

        idx = stokes_map[stokes]
        if idx == 0:  # I
            return ibeam
        return ibeam * self.mueller_matrix[idx, 0] / self.mueller_matrix[0, 0]

    @property
    @abstractmethod
    def I(self):
        """Return I beam in default coordinate system.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def Q(self):
        """Return Q beam in default coordinate system.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def U(self):
        """Return U beam in default coordinate system.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def V(self):
        """Return V beam in default coordinate system.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def beam_area(self, resolution=0.01, rmax=5.0):
        """Return the beam area by numerically integrating the Stokes I beam.

        Parameters
        ----------
        resolution : float, optional
            resolution of numerical integration, by default 0.01
        rmax : float, optional
            Maximum radius for numerical integration, by default 5.0

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError