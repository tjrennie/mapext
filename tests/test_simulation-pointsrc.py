import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy_healpix import HEALPix

from mapext.simulation.pointsrc import pointSource


@pytest.fixture
def wcs_fixture():
    w = WCS(naxis=2)
    w.wcs.crval = [0, 0]
    w.wcs.cdelt = [1 / 60, 1 / 60]
    w.wcs.crpix = [512, 304]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    shape = (768, 1024)
    return w, shape


@pytest.fixture
def hpx_fixture():
    return HEALPix(nside=256, order="ring", frame="galactic")


class TestPointSource:
    def test_point_source_wcs_single_stokes(self, wcs_fixture):
        comp = pointSource()
        comp.projection = wcs_fixture

        x, y = 512, 304
        intensity = 10.0

        outputs = comp.run_simulation((x, y), I=intensity)

        assert isinstance(outputs, dict)
        assert "I" in outputs
        source_map = outputs["I"]

        assert source_map.shape == wcs_fixture[1]
        assert np.isclose(source_map[y, x], intensity)

        source_map[y, x] = 0
        assert np.all(source_map == 0)

    def test_convolved_point_source_wcs_single_stokes(self, wcs_fixture):
        comp = pointSource()
        comp.projection = wcs_fixture

        x, y = 512, 304
        intensity = 10.0

        outputs = comp.run_simulation((x, y), I=intensity, fwhm_deg=1 / 20)

        assert isinstance(outputs, dict)
        assert "I" in outputs
        source_map = outputs["I"]

        assert source_map.shape == wcs_fixture[1]
        assert np.isclose(source_map[y, x], intensity)

    def test_point_source_wcs_all_stokes(self, wcs_fixture):
        comp = pointSource()
        comp.projection = wcs_fixture

        x, y = 512, 304
        intensities = {"I": 1, "Q": 2, "U": 3, "V": 0, "P": 5, "A": 6, "PF": 7}

        outputs = comp.run_simulation((x, y), **intensities)

        assert set(outputs.keys()) == set(intensities.keys())
        for comp_name, val in intensities.items():
            assert np.isclose(outputs[comp_name][y, x], val)

    def test_point_source_wcs_skycoord(self, wcs_fixture):
        comp = pointSource()
        comp.projection = wcs_fixture

        sky_coord = SkyCoord(ra=1.0, dec=2.0, unit="deg")
        intensity = 10.0

        outputs = comp.run_simulation(sky_coord, Q=intensity)
        x, y = wcs_fixture[0].world_to_pixel(sky_coord)

        assert np.isclose(outputs["Q"][int(y), int(x)], intensity)

    def test_out_of_bounds_wcs(self, wcs_fixture):
        comp = pointSource()
        comp.projection = wcs_fixture

        with pytest.raises(ValueError):
            comp.run_simulation((-2, -1000), I=10.0)

    def test_point_source_hpx_single_stokes(self, hpx_fixture):
        comp = pointSource()
        comp.projection = hpx_fixture

        theta = np.radians(45)
        phi = np.radians(60)
        intensity = 10.0

        outputs = comp.run_simulation((theta, phi), Q=intensity)

        assert "Q" in outputs
        source_map = outputs["Q"]

        assert source_map.size == hpx_fixture.npix

        pix = hp.ang2pix(hpx_fixture.nside, theta, phi)
        assert np.isclose(source_map[pix], intensity, atol=1e-3)

        source_map[pix] = 0
        assert np.allclose(source_map, 0, atol=1e-3)

    def test_convolved_point_source_hpx_single_stokes(self, hpx_fixture):
        comp = pointSource()
        comp.projection = hpx_fixture

        theta = np.radians(45)
        phi = np.radians(60)
        intensity = 10.0

        outputs = comp.run_simulation((theta, phi), Q=intensity, fwhm_deg=1 / 20)

        assert "Q" in outputs
        source_map = outputs["Q"]

        assert source_map.size == hpx_fixture.npix

        pix = hp.ang2pix(hpx_fixture.nside, theta, phi)
        assert np.isclose(source_map[pix], intensity, atol=1e-3)

    def test_point_source_hpx_all_stokes(self, hpx_fixture):
        comp = pointSource()
        comp.projection = hpx_fixture

        theta = np.radians(45)
        phi = np.radians(60)
        intensities = {"I": 1, "Q": 2, "U": 3, "V": 0, "P": 5, "A": 6, "PF": 7}

        outputs = comp.run_simulation((theta, phi), **intensities)
        pix = hp.ang2pix(hpx_fixture.nside, theta, phi)

        assert set(outputs.keys()) == set(intensities.keys())
        for comp_name, val in intensities.items():
            assert np.isclose(outputs[comp_name][pix], val)

    def test_point_source_hpx_skycoord(self, hpx_fixture):
        comp = pointSource()
        comp.projection = hpx_fixture

        sky_coord = SkyCoord(l=60.0, b=45.0, unit="deg", frame="galactic")
        intensity = 10.0

        outputs = comp.run_simulation(sky_coord, U=intensity)

        source_map = outputs["U"]
        theta = np.radians(90.0 - sky_coord.b.value)
        phi = np.radians(sky_coord.l.value)
        pix = hp.ang2pix(hpx_fixture.nside, theta, phi)

        assert np.isclose(source_map[pix], intensity, atol=1e-3)

    def test_out_of_bounds_healpix(self, hpx_fixture):
        comp = pointSource()
        comp.projection = hpx_fixture

        with pytest.raises(ValueError):
            comp.run_simulation((np.radians(180), np.radians(370)), Q=10.0)
