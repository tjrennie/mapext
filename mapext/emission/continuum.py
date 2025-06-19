"""Provide various emission models for astrophysical continuum radiation.

The models include:
- Synchrotron emission
- Free-free emission
- AME lognormal emission
- Thermal dust emission

Each model is implemented as a subclass of FittableEmissionModel with evaluation
and derivative methods for fitting purposes.
"""

import logging

import astropy.constants as astropy_c
import numpy as np
from astropy.modeling import Parameter

from mapext.emission.core import FittableEmissionModel

logger = logging.getLogger(__name__)

__all__ = [
    "ame_lognormal",
    "freeFree",
    "freeFree_7500k",
    "synchrotron_1comp",
    "thermalDust",
]


# Synchrotron emission model
class synchrotron_1comp(FittableEmissionModel):
    r"""Emission model for 1-component synchrotron emission without spectral break or curvature (power law)2.

    Defined as

    .. math::

        S_{\nu}^{\mathrm{sync}} = S_{1} \nu^{\alpha},

    where :math:`S_{1}` is the synchrotron amplitude at 1 GHz and :math:`\alpha` is the spectral index (E.G. Carroll and Ostlie 1996).

    References
    ----------
    Carroll, B. W., & Ostlie, D. A. (1996). *An Introduction to Modern Astrophysics*. Addison-Wesley.
    """

    synch_S1 = Parameter(default=0.01, min=0, description="Synchrotron flux at 1 GHz")
    synch_alp = Parameter(default=-0.7, description="Synchrotron spectral index")

    @staticmethod
    def evaluate(nu, area, synch_S1, synch_alp):
        """Evaluate the emission model.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        synch_S1 : float or numpy.ndarray
            Synchrotron flux at 1GHz
        synch_alp : float or numpy.ndarray
            Synchrotron spectral index

        Returns
        -------
        float or numpy.ndarray
            Evaluated function
        """
        return synch_S1 * (nu**synch_alp)

    @staticmethod
    def fit_deriv(nu, area, synch_S1, synch_alp):
        """Evaluate the first derivitives of emission model with respect to input parameters.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        synch_S1 : float or numpy.ndarray
            Synchrotron flux at 1GHz
        synch_alp : float or numpy.ndarray
            Synchrotron spectral index

        Returns
        -------
        float or numpy.ndarray
            First derivitives with repect to input parameters in order.
        """
        d_synch_S1 = nu**synch_alp
        d_synch_alp = synch_S1 * (nu**synch_alp) * np.log(nu)
        return [d_synch_S1, d_synch_alp]


# Free-free emission model (Draine 2011)
class freeFree_7500k(FittableEmissionModel):
    r"""Emission model for free-free emission with an electron temperature of 7500 K.

    This class represents a version of the Draine (2011) free-free emission model.

    Where the free-free optical depth (:math:`\tau^\mathrm{ff}`) is defined as

    .. math::
        \tau^\mathrm{ff}_\nu = 5.468\times 10^{-2} \cdot T_e^{-\frac{3}{2}} \left[ \frac{\nu}{\mathrm{GHz}} \right]^{-2} \left[\frac{EM}{\mathrm{pc\,cm}^-6}\right]  g^\mathrm{ff}_\nu

    where :math:`T_e` is the electron tempertature (assumed to be 7500 K), :math:`\nu` is defined as the frequency of observation, :math:`\mathrm{EM}` is the emission measure, and :math:`g_\nu^\mathrm{ff}` the Gaunt factor. The Gaunt factor is then defined as

    .. math::
        g^\mathrm{ff}_\nu = \ln\left(\exp\left\{5.90 - \frac{\sqrt{3}}{\pi}\ln\left(\left[ \frac{\nu}{\mathrm{GHz}} \right] \left[\frac{T_e}{10^4\,\mathrm{K}}\right] ^\frac{3}{2}\right)\right\} + 2.71828\right),

    and the brightness temperature (:math:`T_\nu^\mathrm{ff}`) then defined as

    .. math::
        T^\mathrm{ff}_\nu = T_e \left(1-e^{-\tau^\mathrm{ff}_\nu}\right).

    To convert this to Janskys, we then assume the mean is filled by emission and convert using

    .. math::
        S^\mathrm{ff}_\nu = \frac{2k_B\Omega\nu^2}{c^2} T^\mathrm{ff}_\nu,

    where :math:`S^\mathrm{ff}_\nu` is the flux density in Janskys, :math:`\Omega` is the beam solid angle, and all other symbols have their usual meanings.

    References
    ----------
    Draine, B. T. (2011). *Physics of the Interstellar and Intergalactic Medium*. Princeton University Press.
    """

    ff_em = Parameter(default=100, min=0, description="Free-free emission measure")

    @staticmethod
    def evaluate(nu, area, ff_em):
        """Evaluate the emission model.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        ff_em : float
            Free-free emission measure

        Returns
        -------
        float or numpy.ndarray
            Evaluated function
        """
        T_e = 7500
        a = (
            0.366
            * np.power(nu, 0.1)
            * np.power(T_e, -0.15)
            * (np.log(np.divide(4.995e-2, nu)) + 1.5 * np.log(T_e))
        )
        T_ff = (
            8.235e-2
            * a
            * np.power(T_e, -0.35)
            * np.power(nu, -2.1)
            * (1.0 + 0.08)
            * ff_em
        )
        return (
            2.0
            * astropy_c.k_B.value
            * area
            * np.power(np.multiply(nu, 1e9), 2)
            / astropy_c.c.value**2
            * T_ff
            * 1e26
        )

    @staticmethod
    def fit_deriv(nu, area, ff_em):
        """Evaluate the first derivitives of emission model with respect to input parameters.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        ff_em : float
            Free-free emission measure

        Returns
        -------
        float or numpy.ndarray
            First derivitives with repect to input parameters in order.
        """
        T_e = 7500
        a = (
            0.366
            * np.power(nu, 0.1)
            * np.power(T_e, -0.15)
            * (np.log(np.divide(4.995e-2, nu)) + 1.5 * np.log(T_e))
        )

        # The derivative of the free-free emission model with respect to ff_em
        T_ff_derivative_ff_em = 8.235e-2 * a * np.power(T_e, -0.35) * np.power(nu, -2.1)
        d_ff_em = (
            2.0
            * astropy_c.k_B.value
            * area
            * np.power(np.multiply(nu, 1e9) / astropy_c.c.value, 2)
            * T_ff_derivative_ff_em
            * 1e26
        )

        return [d_ff_em]


# Free-free emission model (Draine 2011)
class freeFree(FittableEmissionModel):
    r"""Emission model for free-free emission without assumed electron temperature.

    This class represents a version of the Draine (2011) free-free emission model.

    Where the free-free optical depth (:math:`\tau^\mathrm{ff}`) is defined as

    .. math::
        \tau^\mathrm{ff}_\nu = 5.468\times 10^{-2} \cdot T_e^{-\frac{3}{2}} \left[ \frac{\nu}{\mathrm{GHz}} \right]^{-2} \left[\frac{EM}{\mathrm{pc\,cm}^-6}\right]  g^\mathrm{ff}_\nu

    where :math:`T_e` is the electron tempertature, :math:`\nu` is defined as the frequency of observation, :math:`\mathrm{EM}` is the emission measure, and :math:`g_\nu^\mathrm{ff}` the Gaunt factor. The Gaunt factor is then defined as

    .. math::
        g^\mathrm{ff}_\nu = \ln\left(\exp\left\{5.90 - \frac{\sqrt{3}}{\pi}\ln\left(\left[ \frac{\nu}{\mathrm{GHz}} \right] \left[\frac{T_e}{10^4\,\mathrm{K}}\right] ^\frac{3}{2}\right)\right\} + 2.71828\right),

    and the brightness temperature (:math:`T_\nu^\mathrm{ff}`) then defined as

    .. math::
        T^\mathrm{ff}_\nu = T_e \left(1-e^{-\tau^\mathrm{ff}_\nu}\right).

    To convert this to Janskys, we then assume the mean is filled by emission and convert using

    .. math::
        S^\mathrm{ff}_\nu = \frac{2k_B\Omega\nu^2}{c^2} T^\mathrm{ff}_\nu,

    where :math:`S^\mathrm{ff}_\nu` is the flux density in Janskys, :math:`\Omega` is the beam solid angle, and all other symbols have their usual meanings

    References
    ----------
    Draine, B. T. (2011). *Physics of the Interstellar and Intergalactic Medium*. Princeton University Press.
    """

    ff_em = Parameter(default=100, min=0, description="Free-free emission measure")
    ff_Te = Parameter(default=7500, min=0, description="Free-free electron temperature")

    @staticmethod
    def evaluate(nu, area, ff_em, ff_Te):
        """Evaluate the emission model.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        ff_em : float
            Free-free emission measure
        ff_Te : float
            Free-free electron temperature

        Returns
        -------
        float or numpy.ndarray
            First derivitives with repect to input parameters in order.
        """
        g = np.log(
            np.exp(5.90 - (np.sqrt(3) / np.pi * np.log(nu * ((ff_Te / 1e4) ** 1.5))))
            + 2.71828
        )
        tau = 5.468e-2 * (ff_Te**-1.5) * (nu**-2) * ff_em * g
        T_ff = ff_Te * (1 - np.exp(-1 * tau))
        return (
            2.0
            * astropy_c.k_B.value
            * area
            * np.power(np.multiply(nu, 1e9), 2)
            / astropy_c.c.value**2
            * T_ff
            * 1e26
        )

    @staticmethod
    def fit_deriv(nu, area, ff_em, ff_Te):
        """Evaluate the first derivitives of emission model with respect to input parameters.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        ff_em : float
            Free-free emission measure
        ff_Te : float
            Free-free electron temperature

        Returns
        -------
        float or numpy.ndarray
            First derivitives with repect to input parameters in order.
        """
        g = np.log(
            np.exp(5.90 - (np.sqrt(3) / np.pi * np.log(nu * ((ff_Te / 1e4) ** 1.5))))
            + 2.71828
        )
        tau = 5.468e-2 * (ff_Te**-1.5) * (nu**-2) * ff_em * g
        T_ff = ff_Te * (1 - np.exp(-1 * tau))
        S = (
            2.0
            * astropy_c.k_B.value
            * area
            * np.power(np.multiply(nu, 1e9), 2)
            / astropy_c.c.value**2
            * T_ff
            * 1e26
        )

        dtau_dem = tau / ff_em
        dTff_dem = T_ff * dtau_dem / (np.exp(tau) - 1)
        dS_dem = S * dTff_dem / T_ff

        dg_dTe = 225693 / (
            (ff_Te * ((nu * (ff_Te**1.5)) ** (np.sqrt(3) / np.pi))) + (272908 * ff_Te)
        )
        dtau_dTe = (tau * dg_dTe / g) - (tau * 1.5 / ff_Te)
        dTff_dTe = (T_ff * dtau_dTe / (np.exp(tau) - 1)) + T_ff / ff_Te
        dS_dTe = S * dTff_dTe / T_ff

        return [dS_dem, dS_dTe]


# AME lognormal
class ame_lognormal(FittableEmissionModel):
    r"""Emission model for an AME lognormal source.

    The AME flux density (:math:`S^{\mathrm{AME}}_\nu`) is defined as

    .. math::
        S^{\mathrm{AME}}_{\nu} = A_{\mathrm{AME}} \cdot \exp\left\{ -\frac{1}{2}\left( \frac{\ln(\nu / \nu_{\mathrm{AME}})}{W_{\mathrm{AME}}} \right)^2  \right\},

    where :math:`A_\mathrm{AME}` is the AME amplitude, :math:`\nu_\mathrm{AME}` is the AME peak frequency, and :math:`W_\mathrm{AME}` is the full-width half maximum in log-space (e.g. Stevenson 2014).

    References
    ----------
    Stevenson, M. A. (2014). *The anomalous microwave emission*. PhD thesis, University of Manchester.
    """

    ame_ampl = Parameter(default=0.1, min=0, description="AME peak flux density")
    ame_peak = Parameter(default=27, min=5, max=60, description="AME peak frequency")
    ame_width = Parameter(default=0.5, min=0, max=1, description="AME lognormal width")

    @staticmethod
    def evaluate(nu, area, ame_ampl, ame_peak, ame_width):
        """Evaluate the emission model.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        ame_ampl : float
            AME peak flux density
        ame_peak : float
            AME peak frequency
        ame_width : float
            AME lognormal width

        Returns
        -------
        float or numpy.ndarray
            Evaluated function
        """
        nlog = np.log(nu)
        nmaxlog = np.log(ame_peak)
        return ame_ampl * np.exp(-0.5 * ((nlog - nmaxlog) / ame_width) ** 2)

    @staticmethod
    def fit_deriv(nu, area, ame_ampl, ame_peak, ame_width):
        """Evaluate the first derivitives of emission model with respect to input parameters.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        ame_ampl : float
            AME peak flux density
        ame_peak : float
            AME peak frequency
        ame_width : float
            AME lognormal width

        Returns
        -------
        list
            First derivitives with repect to input parameters in order.
        """

        def evaluate2(nu, area, ame_ampl, ame_peak, ame_width):
            nlog = np.log(nu)
            nmaxlog = np.log(ame_peak)
            return ame_ampl * np.exp(-0.5 * ((nlog - nmaxlog) / ame_width) ** 2)

        S = evaluate2(nu, area, ame_ampl, ame_peak, ame_width)
        d_ame_ampl = S / ame_ampl
        d_ame_peak = (
            S * -1 * ((np.log(ame_peak) - np.log(nu)) / (ame_peak * (ame_width**2)))
        )
        d_ame_width = S * ((np.log(ame_peak) - np.log(nu)) ** 2 / (ame_width**3))
        return [d_ame_ampl, d_ame_peak, d_ame_width]


# Thermal dust
class thermalDust(FittableEmissionModel):
    r"""Emission model for the Planck modified thermal dust curve - a modified blackbody with opacity varying as frequency to some dust spectral index.

    The thermal dust flux denisty (:math:`S^\mathrm{td}_\nu`) is defined as

    .. math::
        S^\mathrm{td}_\nu = \frac{2k_B\Omega\nu^3}{c^2} \frac{1}{e^{h\nu/k_BT_b}-1} \cdot \tau_{\nu_0} \cdot \left(\frac{\nu}{\nu_0}\right)^\beta,

    where :math:`\Omega` is the beam solid angle, :math:`\tau_{\nu_0}` is the dust optical depth at frequency :math:`\nu_0` (set here to 353GHz um), :math:`\beta` being the dust spectral index and :math:`T_d` the dust temperature.

    References
    ----------
    Draine, B. T., & Li, A. (2001). Infrared Emission from Interstellar Dust. I. Stochastic Heating of Small Grains. *Astrophysical Journal*, 551(2), 807-824. https://doi.org/10.1086/320227
    Draine, B. T. (2011). *Physics of the Interstellar and Intergalactic Medium*. Princeton University Press.
    """

    tdust_Td = Parameter(default=20, min=0, description="Thermal dust temperature")
    tdust_tau = Parameter(
        default=-4, description="Thermal dust opacity (given as log_10(tau))"
    )
    tdust_beta = Parameter(default=1.5, description="thermal dust spectral index")

    @staticmethod
    def evaluate(nu, area, tdust_Td, tdust_tau, tdust_beta):
        """Evaluate the emission model.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        tdust_Td : float
            Thermal dust temperature
        tdust_tau : float
            Thermal dust opacity (given as log_10(tau))
        tdust_beta : float
            Thermal dust spectral index

        Return
        ------
        float or numpy.ndarray
            Evaluated function
        """
        nu9 = np.multiply(nu, 1e9)
        planck = np.exp(astropy_c.h.value * nu9 / astropy_c.k_B.value / tdust_Td) - 1.0
        modify = 10**tdust_tau * (nu9 / 1.2e12) ** tdust_beta
        return (
            2
            * astropy_c.h.value
            * nu9**3
            / astropy_c.c.value**2
            / planck
            * modify
            * area
            * 1e26
        )

    @staticmethod
    def fit_deriv(nu, area, tdust_Td, tdust_tau, tdust_beta):
        """Evaluate the first derivatives of emission model with respect to input parameters.

        Parameters
        ----------
        nu : float or numpy.ndarray
            Frequency in GHz
        area : float or numpy.ndarray
            Beam area in steradians
        tdust_Td : float
            Thermal dust temperature
        tdust_tau : float
            Thermal dust opacity (given as log_10(tau))
        tdust_beta : float
            Thermal dust spectral index

        Return
        ------
        list
            First derivatives with respect to input parameters in the order:
            [dS/dTd, dS/dtau, dS/dbeta]
        """
        # Convert frequency to Hz
        nu9 = np.multiply(nu, 1e9)

        # Planck function and modify term for dust opacity and spectral index
        planck = np.exp(astropy_c.h.value * nu9 / astropy_c.k_B.value / tdust_Td) - 1.0
        modify = 10**tdust_tau * (nu9 / 1.2e12) ** tdust_beta

        # Evaluate the thermal dust flux density S
        S = (
            2
            * astropy_c.h.value
            * nu9**3
            / astropy_c.c.value**2
            / planck
            * modify
            * area
            * 1e26
        )

        # Derivative with respect to Td (dust temperature)
        hvkT = astropy_c.h.value * nu9 / (astropy_c.k_B.value * tdust_Td)
        d_tdust_Td = S * (hvkT) * (1 / tdust_Td) * (np.exp(hvkT) / (np.exp(hvkT) - 1))

        # Derivative with respect to tau (dust opacity)
        d_tdust_tau = S * np.log(10)

        # Derivative with respect to beta (dust spectral index)
        d_tdust_beta = -S * np.log(S * (nu9 / 353e9) ** (1 - tdust_beta))

        return [d_tdust_Td, d_tdust_tau, d_tdust_beta]
