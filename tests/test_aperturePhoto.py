import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.wcs import WCS

from mapext.photometry.aperturephotometry import bullseye
from mapext.simulation import stokesMapSimulation
from mapext.simulation.pointsrc import pointSource
from mapext.simulation.noise import whiteNoise
from mapext.core.source import astroSrc


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
        whiteNoise(I_rms=0.2, Q_rms=0.2, U_rms=0.2)
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

    print(bullseye(
        point_source_with_background_wcs,
        s
    ))

    assert False