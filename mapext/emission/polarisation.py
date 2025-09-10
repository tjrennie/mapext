"""Define a constant polarisation model for emission.

Include the `constantPol` class, which models polarisation using
parameters such as polarisation fraction and polarisation angle.
"""

import logging

import numpy as np
from astropy.modeling import Parameter

from mapext.core.stokes import StokesComp, wrap_theta
from mapext.emission.core import FittablePolarisationModel

logger = logging.getLogger(__name__)


class constantPol(FittablePolarisationModel):
    r"""Contant polarisation model, defined by p[olarisation fraction (PF) and polarisation angle (PolAngle).

    Polarisation components are defined as

        .. math::
            \\begin{bmatrix} I \\\ Q \\\ U \\\ V \\end{bmatrix} =
            I_\\mathrm{model}\\begin{bmatrix} 1 \\\ \\mathrm{PF}\\cos(2\\gamma) \\\ \\mathrm{PF}\\sin{2\\gamma} \\\ 0 \\end{bmatrix}.

    """

    pol_PF = Parameter(default=0.5, min=0, description="Polarisation fraction")
    pol_Angle = Parameter(default=np.pi / 2, description="Polarisation angle")

    @staticmethod
    def evaluate(nu, area, stokes, pol_PF, pol_Angle):
        """Evaluate the emission model.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        stokes : int or numpy.ndarray
            Stokes component(s) to evaluate
        pol_PF : float
            Polarisation fraction
        pol_Angle : float
            Polarisation angle

        Returns
        -------
        float or numpy.ndarray
            Evaluated function
        """
        stokes = np.array(stokes, dtype=int)

        # Create output array and fill with NaNs
        outarray = np.full(nu.shape, np.nan)
        # Calculate Stokes' parameters
        outarray[stokes == int(StokesComp("I"))] = 1
        outarray[stokes == int(StokesComp("Q"))] = pol_PF * np.cos(2 * pol_Angle)
        outarray[stokes == int(StokesComp("U"))] = pol_PF * np.sin(2 * pol_Angle)
        outarray[stokes == int(StokesComp("V"))] = 0
        # Calculate derived Stokes' parameters
        outarray[stokes == int(StokesComp("P"))] = pol_PF
        outarray[stokes == int(StokesComp("A"))] = wrap_theta(pol_Angle)
        outarray[stokes == int(StokesComp("PF"))] = pol_PF

        return outarray

    @staticmethod
    def fit_deriv(nu, area, stokes, pol_PF, pol_Angle):
        """Evaluate the first derivitives of emission model with respect to input parameters.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        stokes : int or numpy.ndarray
            Stokes component(s) to evaluate
        pol_PF : float
            Polarisation fraction
        pol_Angle : float
            Polarisation angle

        Return
        ------
        list
            First derivitives with repect to input parameters in order.
        """
        nu, _beam, stokes = np.broadcast_arrays(nu, area, stokes)
        d_pol_PF, d_pol_Angle = np.full(nu.shape, np.nan), np.full(nu.shape, np.nan)

        # I
        d_pol_PF[stokes == int(StokesComp("I"))] = 0
        d_pol_Angle[stokes == int(StokesComp("I"))] = 0
        # Q
        d_pol_PF[stokes == int(StokesComp("Q"))] = np.cos(2 * pol_Angle)
        d_pol_Angle[stokes == int(StokesComp("Q"))] = (
            -pol_PF * 2 * pol_Angle * np.sin(2 * pol_Angle)
        )
        # U
        d_pol_PF[stokes == int(StokesComp("U"))] = np.sin(2 * pol_Angle)
        d_pol_Angle[stokes == int(StokesComp("U"))] = (
            pol_PF * 2 * pol_Angle * np.cos(2 * pol_Angle)
        )
        # V
        d_pol_PF[stokes == int(StokesComp("V"))] = 0
        d_pol_Angle[stokes == int(StokesComp("V"))] = 0
        # P
        d_pol_PF[stokes == int(StokesComp("P"))] = 1
        d_pol_Angle[stokes == int(StokesComp("P"))] = 0
        # A
        d_pol_PF[stokes == int(StokesComp("A"))] = 0
        d_pol_Angle[stokes == int(StokesComp("A"))] = 1
        # PF
        d_pol_PF[stokes == int(StokesComp("PF"))] = 1
        d_pol_Angle[stokes == int(StokesComp("PF"))] = 0

        return [d_pol_PF, d_pol_Angle]
