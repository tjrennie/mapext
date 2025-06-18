import logging

import numpy as np
import pytest
from astropy.coordinates import Galactic
from astropy.wcs import WCS
from astropy_healpix import HEALPix

from mapext.simulation import stokesMapSimulation, stokesMapSimulationComponent


# Dummy subclass to override run_simulation
class DummyStokesComponent(stokesMapSimulationComponent):
    def run_simulation(self, test_param=10):
        shape = self.shape or (5,)
        return {
            "I": np.ones(shape),
            "Q": np.zeros(shape),
            "U": np.full(shape, 1.0),
            "V": np.full(shape, 0.0),
            "P": np.full(shape, 1.0),
            "A": np.full(shape, 1.0),
            "PF": np.full(shape, 1.0),
        }


# Test class for stokesMapSimulationComponent
class TestStokesMapSimulationComponent:
    def test_repr_str(self):
        DummyStokesComponent._default_simulation_params = {"test_param": 0}
        comp = DummyStokesComponent(test_param=42)
        assert isinstance(repr(comp), str)
        assert isinstance(str(comp), str)


    def test_unexpected_param_logging(self):
        with pytest.raises(ValueError, match="Unexpected simulation parameter: 'bogus' will be ignored."):
            DummyStokesComponent(bogus=123)

    def test_set_get_simulation_param(self):
        comp = DummyStokesComponent()
        comp._simulation_parameters["test_param"] = 123
        assert comp.test_param == 123
        comp.test_param = 456
        assert comp.test_param == 456

    def test_projection_healpix(self):
        hp = HEALPix(nside=4, order="ring", frame=Galactic())
        comp = DummyStokesComponent()
        comp.projection = hp
        assert comp._shape_out == hp.npix

    def test_projection_wcs(self):
        wcs = WCS(naxis=2)
        shape = (10, 10)
        comp = DummyStokesComponent()
        comp.projection = (wcs, shape)
        assert comp._shape_out == shape

    def test_invalid_projection(self):
        comp = DummyStokesComponent()
        with pytest.raises(ValueError):
            comp.projection = "not a projection"

    def test_stokes_properties(self):
        shape = (4,)
        comp = DummyStokesComponent()
        comp.shape = shape
        comp.generate_simulation()

        assert np.array_equal(comp.I, np.ones(shape))
        assert np.array_equal(comp.Q, np.zeros(shape))
        assert np.array_equal(comp.U, np.full(shape, 1.0))
        assert np.array_equal(comp.V, np.full(shape, 0.0))
        assert np.array_equal(comp.P, np.full(shape, 1.0))
        assert np.array_equal(comp.A, np.full(shape, 1.0))
        assert np.array_equal(comp.PF, np.full(shape, 1.0))

    def test_reset_cached_maps(self):
        comp = DummyStokesComponent()
        comp._maps_cached = True
        comp._I_MAP = np.array([1, 2, 3])
        comp.reset_cached_maps()
        assert comp._I_MAP is None
        assert not comp._maps_cached


# Test class for stokesMapSimulation
class TestStokesMapSimulation:
    def test_add_simulation_component(self):
        comp = DummyStokesComponent()
        sim = stokesMapSimulation()
        sim.add_simulation_component(comp)
        assert len(sim.simulation_components) == 1

    def test_add_invalid_component(self):
        sim = stokesMapSimulation()
        with pytest.raises(TypeError):
            sim.add_simulation_component("InvalidComponent")

    def test_update_component_projections(self):
        sim = stokesMapSimulation()
        sim.add_simulation_component(DummyStokesComponent())
        sim.add_simulation_component(DummyStokesComponent())

        hp = HEALPix(nside=4, order="ring", frame=Galactic())
        sim._projection = hp
        sim._shape = (hp.npix,)

        sim.update_component_projections()

        assert sim.simulation_components[0].projection == hp
        assert sim.simulation_components[1].projection == hp

    def test_projection_property_setter(self):
        sim = stokesMapSimulation()
        sim.add_simulation_component(DummyStokesComponent())
        sim.add_simulation_component(DummyStokesComponent())

        hp = HEALPix(nside=4, order="ring", frame=Galactic())
        sim.set_projection(hp)
        assert sim.projection == hp
        assert sim.simulation_components[0].projection == hp
        assert sim.simulation_components[1].projection == hp

    def test_invalid_projection_property_setter(self):
        sim = stokesMapSimulation()
        with pytest.raises(ValueError):
            sim.projection = "InvalidProjection"

    def test_set_projection(self):
        comp = DummyStokesComponent()
        sim = stokesMapSimulation()
        sim.add_simulation_component(comp)

        hp = HEALPix(nside=4, order="ring", frame=Galactic())
        sim.set_projection(hp)

        assert comp.projection == hp

    def test_get_combined_stokes_map(self):
        comp1 = DummyStokesComponent()
        comp2 = DummyStokesComponent()
        sim = stokesMapSimulation()
        sim.add_simulation_component(comp1)
        sim.add_simulation_component(comp2)
        hp = HEALPix(nside=4, order="ring", frame=Galactic())
        sim.set_projection(hp)

        # Generate combined Stokes I map (just testing the method without checking actual values)
        combined_map = sim.I
        print(
            sim.shape,
            sim.simulation_components[0].shape,
            sim.simulation_components[1].shape,
        )
        print(
            sim.projection,
            sim.simulation_components[0].projection,
            sim.simulation_components[1].projection,
        )
        assert combined_map is not None
        assert combined_map.shape == hp.npix

    def test_invalid_stokes_type(self):
        comp = DummyStokesComponent()
        sim = stokesMapSimulation()
        sim.add_simulation_component(comp)

        with pytest.raises(ValueError):
            sim._get_stokes_map("InvalidType")
