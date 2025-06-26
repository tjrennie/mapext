import pytest
from astropy.wcs import WCS

from mapext.core.source import astroSrc
from mapext.photometry.aperturephotometry import apertureAnnulus
from mapext.simulation import stokesMapSimulation
from mapext.simulation.noise import coloredNoise, whiteNoise
from mapext.simulation.pointsrc import pointSource


@pytest.fixture
def point_source_with_background_wcs():
    w = WCS(naxis=2)
    w.wcs.crval = [0, 0]
    w.wcs.cdelt = [1 / 60, 1 / 60]
    w.wcs.crpix = [32, 32]
    w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    shape = (63, 63)

    simmap = stokesMapSimulation()
    simmap.add_simulation_component(
        coloredNoise(alpha=-1, I_rms=0.1, Q_rms=0.1, U_rms=0.1)
    )
    simmap.add_simulation_component(
        whiteNoise(I_rms=0.1, Q_rms=0.1, U_rms=0.1)
    )
    simmap.add_simulation_component(
        pointSource(I=1, Q=0.4, U=0.5, fwhm_deg=10/60)
    )
    simmap.set_projection((w, shape))
    return simmap

def test_1(point_source_with_background_wcs):

    s = astroSrc(
        name="Test Source",
        coords=(0,0),
        frame="galactic"
    )

    apertureAnnulus(
        point_source_with_background_wcs,
        s
    )
