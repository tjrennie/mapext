import astropy.units as astropy_u
import numpy as np
import pytest

from mapext.core.map import stokesMap


def test_stokesmap_init():
    smap = stokesMap()
    assert smap.assume_v_0 is True
    assert smap._maps_cached == []


def test_load_from_kwargs():
    smap = stokesMap(I=np.array([1, 2, 3]), Q=np.array([0.5, 0.2, 0.1]))
    assert "I" in smap._maps_cached
    assert "Q" in smap._maps_cached
    assert np.array_equal(smap._I, np.array([1, 2, 3]))
    assert np.array_equal(smap._Q, np.array([0.5, 0.2, 0.1]))


def test_frequency_setter():
    smap = stokesMap()
    smap.frequency = 1.4e9  # GHz
    assert smap._frequency.unit == astropy_u.Hz
    assert np.isclose(smap._frequency.value, 1.4e9)


def test_wavelength_setter():
    smap = stokesMap()
    smap.wavelength = 0.21  # meters
    print(smap.wavelength)
    assert smap._wavelength.unit == astropy_u.m
    assert np.isclose(smap._wavelength.value, 0.21)


def test_pol_convention():
    smap = stokesMap()
    smap.pol_convention = "IAU"
    assert smap.pol_convention == "IAU"
    with pytest.raises(ValueError):
        smap.pol_convention = "INVALID"


def test_switch_pol_conventions():
    smap = stokesMap(U=np.array([1, -1, 2]), A=np.array([30, 45, 60]))
    smap.pol_convention = "COSMO"
    smap.switch_pol_conventions()
    assert smap.pol_convention == "IAU"
    assert np.array_equal(smap._U, np.array([-1, 1, -2]))
    assert np.array_equal(smap._A, np.array([-30, -45, -60]))
