import numpy as np
import pytest

from mapext.core.stokes import (
    StokesComp,
    get_derivative_function,
    get_stokes_value_mapping,
    wrap_theta,
)


def test_wrap_theta():
    assert wrap_theta(0) == 0
    assert wrap_theta(np.pi) == 0
    assert wrap_theta(-np.pi) == 0
    assert wrap_theta(np.pi / 2) == np.pi / 2
    assert wrap_theta(-np.pi / 2) == np.pi / 2


def test_stokescomp_init():
    assert StokesComp("I").letter == "I"
    assert StokesComp("q").letter == "Q"
    with pytest.raises(ValueError):
        StokesComp("X")


def test_stokescomp_conversion():
    s = StokesComp("Q")
    assert int(s) == 1
    assert float(s) == 1.0


def test_stokescomp_repr():
    s = StokesComp("U")
    assert repr(s) == "Stokes('U')"


def test_get_stokes_value_mapping():
    func = get_stokes_value_mapping("I", ["I"])
    assert func(I=10) == 10
    with pytest.raises(ValueError):
        func()


def test_get_derivative_function():
    def sample_func(x):
        return x**2

    derivative_func = get_derivative_function(sample_func, derivative=True)
    result = derivative_func(x=2)
    expected_derivative = 4  # Approximate numerical derivative of x^2 at x=2
    assert np.isclose(result["x"], expected_derivative, atol=1e-5)
