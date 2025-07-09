import numpy as np
import pytest
from astropy.wcs import WCS

from mapext.core.source import astroSrc
from mapext.photometry.aperturephotometry import apertureAnnulus
from mapext.simulation import stokesMapSimulation
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
    # simmap.add_simulation_component(
    #     coloredNoise(alpha=-1, I_rms=0.1, Q_rms=0.1, U_rms=0.1)
    # )
    # simmap.add_simulation_component(whiteNoise(I_rms=0.1, Q_rms=0.1, U_rms=0.1))
    simmap.add_simulation_component(
        pointSource(I=10 / 4.5, Q=4 / 4.5, U=2 / 4.5, fwhm_deg=2 / 60)
    )
    simmap.set_projection((w, shape))
    simmap.I
    return simmap


def test_photometry_apertureannulus(point_source_with_background_wcs):

    s = astroSrc(name="Test Source", coords=(0, 0), frame="galactic")

    print("=" * 80)

    res = apertureAnnulus(
        point_source_with_background_wcs,
        s,
        result_to_src=False,
        return_results=True,
        plot=False,
    )

    print("=" * 80)
    for _, __ in zip(res[0], res[2]):
        a = np.nansum(getattr(point_source_with_background_wcs, __))
        print(_, a, _ / a)
        if _ == "I":
            assert _ - 10 < 0.1
        elif _ == "Q":
            assert _ - 4 < 0.1
        elif _ == "U":
            assert _ - 2 < 0.1
