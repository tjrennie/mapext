import numpy as np
import pytest
from astropy.wcs import WCS
from astropy_healpix import HEALPix

from mapext.simulation.noise import coloredNoise, whiteNoise


@pytest.fixture
def wcs_fixture():
    """Create a WCS object for testing."""
    w = WCS(naxis=2)
    w.wcs.crval = [0, 0]
    w.wcs.cdelt = [1 / 60, 1 / 60]
    w.wcs.crpix = [512, 304]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    shape = (768, 1024)
    return w, shape


@pytest.fixture
def hpx_fixture():
    """Create a HEALPix object for testing."""
    return HEALPix(nside=256, order="ring", frame="galactic")


class TestWhiteNoise:

    def test_white_noise_wcs(self, wcs_fixture):
        comp = whiteNoise(I_rms=5, Q_rms=3, U_rms=2)
        comp.projection = wcs_fixture
        # Assert shapes must be correct
        assert comp.I.shape == wcs_fixture[-1]
        assert comp.P.shape == wcs_fixture[-1]
        # Assert distributions requested are correct
        assert np.abs(np.nanstd(comp.I) - 5) < 1e-2
        assert np.abs(np.nanstd(comp.Q) - 3) < 1e-2
        assert np.abs(np.nanstd(comp.U) - 2) < 1e-2

    def test_white_noise_hpx(self, hpx_fixture):
        comp = whiteNoise(I_rms=5, Q_rms=3, U_rms=2)
        comp.projection = hpx_fixture
        # Assert shapes must be correct
        assert comp.I.shape == hpx_fixture.npix
        assert comp.P.shape == hpx_fixture.npix
        # Assert distributions requested are correct
        assert np.abs(np.nanstd(comp.I) - 5) < 1e-2
        assert np.abs(np.nanstd(comp.Q) - 3) < 1e-2
        assert np.abs(np.nanstd(comp.U) - 2) < 1e-2


class TestColoredNoise:

    def test_colored_noise_wcs(self, wcs_fixture):
        for col in [-4, -2, -1, 0, 1, 2, 4]:
            comp = coloredNoise(alpha=col, I_rms=5, Q_rms=3, U_rms=2)
            comp.projection = wcs_fixture
            # Assert shapes must be correct
            assert comp.I.shape == wcs_fixture[-1]
            assert comp.P.shape == wcs_fixture[-1]
            # Assert distributions requested are correct
            assert np.abs(np.nanstd(comp.I) - 5) < 1e-2
            assert np.abs(np.nanstd(comp.Q) - 3) < 1e-2
            assert np.abs(np.nanstd(comp.U) - 2) < 1e-2

    def test_colored_noise_hpx(self, hpx_fixture):
        for col in [-4, -2, -1, 0, 1, 2, 4]:
            comp = coloredNoise(alpha=col, I_rms=5, Q_rms=3, U_rms=2)
            comp.projection = hpx_fixture
            # Assert shapes must be correct
            assert comp.I.shape == hpx_fixture.npix
            assert comp.P.shape == hpx_fixture.npix
            # Assert distributions requested are correct
            assert np.abs(np.nanstd(comp.I) - 5) < 1e-2
            assert np.abs(np.nanstd(comp.Q) - 3) < 1e-2
            assert np.abs(np.nanstd(comp.U) - 2) < 1e-2
