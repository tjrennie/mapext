import numpy as np
import pytest
from astropy.modeling import Parameter

from mapext.emission.core import (
    FittableEmissionModel,
    FittablePolarisationModel,
    compoundFittablePolarisedEmissionModel,
    fittablePolarisedEmissionModel,
)


class TestEmissionModel(FittableEmissionModel):
    param1 = Parameter(default=1.0)

    @staticmethod
    def evaluate(nu, beam, param1):
        return param1 * nu * beam


class TestPolarisationModel(FittablePolarisationModel):
    param1 = Parameter(default=1.0)

    @staticmethod
    def evaluate(nu, beam, stokes, param1):
        return param1 * np.sin(nu) * np.cos(beam) * stokes


@pytest.fixture
def emission_model():
    return TestEmissionModel()


@pytest.fixture
def polarisation_model():
    return TestPolarisationModel()


@pytest.fixture
def polarised_emission_model(emission_model, polarisation_model):
    return fittablePolarisedEmissionModel(emission_model, polarisation_model)


@pytest.fixture
def compound_model(polarised_emission_model):
    return polarised_emission_model + polarised_emission_model


def test_emission_model(emission_model):
    assert isinstance(emission_model, FittableEmissionModel)
    result = emission_model(2.0, 3.0)
    assert np.isclose(result, 6.0)


def test_polarisation_model(polarisation_model):
    assert isinstance(polarisation_model, FittablePolarisationModel)
    result = polarisation_model(np.pi / 2, 0.0, 1.0)
    assert np.isclose(result, 1.0)


def test_fittable_polarised_emission_model(polarised_emission_model):
    assert isinstance(polarised_emission_model, fittablePolarisedEmissionModel)


def test_compound_model(compound_model):
    assert isinstance(compound_model, compoundFittablePolarisedEmissionModel)


def test_invalid_model_set_axis():
    with pytest.raises(ValueError):
        fittablePolarisedEmissionModel(
            TestEmissionModel(), TestPolarisationModel(model_set_axis=1)
        )
