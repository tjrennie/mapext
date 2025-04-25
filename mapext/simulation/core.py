"""Provide the core functionality for simulating Stokes parameter maps.

Include the `stokesMapSimulationComponent` class, which allows for the generation
and management of simulated Stokes parameter maps (I, Q, U, V, P, A, PF) using
specified parameters and projections.
"""

import inspect
import warnings

import numpy as np
from astropy.wcs import WCS
from astropy_healpix import HEALPix

from mapext.core.map import stokesMap
from mapext.core.stokes import get_stokes_value_mapping, wrap_theta

__all__ = [
    "stokesMapSimulation",
    "stokesMapSimulationComponent",
]


class stokesMapSimulationComponent:
    """A component for simulating Stokes parameter maps.

    This class provides methods to simulate and manage
    Stokes parameter maps (I, Q, U, V, P, A, PF) using specified parameters
    and projections.

    Parameters
    ----------
    assume_v_0 : bool, optional
        Whether to assume the V Stokes parameter is zero. Default is True.
    **kwargs : dict
        Keyword arguments corresponding to simulation parameters.
    """

    def __init__(self, assume_v_0: bool = True, **kwargs):
        """Class to hold simulation components to be used in simulated maps."""
        self.params_supplied = []
        self.assume_v_0 = assume_v_0
        self._projection = None
        self._shape_out = None
        self.reset_cached_maps()
        self._simulation_parameters = {}

        # Capture expected parameter keys
        expected_keys = list(self._default_simulation_params.keys())

        # Handle unexpected keys
        for key in kwargs.keys():
            if key not in expected_keys:
                warnings.warn(
                    f"Unexpected simulation parameter: '{key}' will be ignored."
                )

        # Initialize known parameters
        for key in expected_keys:
            self._simulation_parameters[key] = kwargs.get(
                key, self._default_simulation_params[key]
            )

    def __repr__(self) -> str:
        repr_str = f"<{self.__class__.__name__}("
        app = ""
        for key, val in self._simulation_parameters.items():
            repr_str += f"{app}{key}={val}"
            app = ", "
        repr_str += ")>\n"
        return repr_str

    def __str__(self) -> str:
        lines = [f"Component: {self.__class__.__name__}"]

        # Projection and map info
        proj_type = type(self._projection).__name__ if self._projection else "None"
        lines.append(f"Projection type: {proj_type}")
        lines.append(f"Output map shape: {self._shape_out}")
        lines.append(f"Maps cached: {self._maps_cached}")

        # Parameter summary
        lines.append("Parameters:")
        param_names = list(self._simulation_parameters.keys())
        param_values = list(self._simulation_parameters.values())

        if not param_names:
            lines.append("    (none)")
        else:
            # Compute column widths individually
            widths = [
                max(len(name), len(f"{val:}"))
                for name, val in zip(param_names, param_values)
            ]

            # Build each line with a single space between columns
            name_line = "    " + " ".join(
                f"{param_names[i]:^{widths[i]}}" for i in range(len(param_names))
            )
            dash_line = "    " + " ".join(
                "-" * widths[i] for i in range(len(param_names))
            )
            value_line = "    " + " ".join(
                f"{param_values[i]:>{widths[i]}}" for i in range(len(param_names))
            )

            lines.extend([name_line, dash_line, value_line, ""])

        return "\n".join(lines)

    def __setattr__(self, name, value):
        """Set an attribute of the class.

        If the attribute is a simulation parameter, update it and reset cached maps.

        Parameters
        ----------
        name : str
            The name of the attribute to set.
        value : any
            The value to set the attribute to.
        """
        if (
            "_simulation_parameters" in self.__dict__
            and name in self._simulation_parameters
        ):
            if self._simulation_parameters.get(name) != value:
                self._simulation_parameters[name] = value
                self.reset_cached_maps()
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        """Get an attribute of the class.

        If the attribute is a simulation parameter, return its value.
        If the attribute is assume_v_0, return its value.
        Otherwise, use the default behavior.

        Parameters
        ----------
        name : str
            The name of the attribute to get.

        Returns
        -------
        any
            The value of the attribute.
        """
        if name in self._simulation_parameters:
            return self._simulation_parameters[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def reset_cached_maps(self):
        """Reset cached maps to None."""
        self._maps_cached = False
        self._I_MAP = None
        self._Q_MAP = None
        self._U_MAP = None
        self._V_MAP = None
        self._P_MAP = None
        self._A_MAP = None
        self._PF_MAP = None

    @property
    def _default_simulation_params(self):
        sig = inspect.signature(self.run_simulation)
        return {
            name: param.default
            for name, param in sig.parameters.items()
            if name != "self"
            and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            and param.default is not param.empty
        }

    @property
    def shape(self):
        """Shape of the map data array.

        Returns
        -------
        tuple
            Shape of the map data array.
        """
        return self._shape_out

    @shape.setter
    def shape(self, shape):
        """Set the shape of the map data array.

        Parameters
        ----------
        shape : tuple
            Shape of the map data array.
        """
        self._shape_out = np.array(shape, dtype=int)
        self._projection = None

    @property
    def projection(self):
        """Simulation output projection.

        Returns
        -------
        WCSobject or astropy HEALPix object
            Projection object for the simulation output.
        """
        return self._projection

    @projection.setter
    def projection(self, projection):
        """Set the projection object for the simulation output.

        Parameters
        ----------
        projection: (WCS, np.ndarray) or HEALPix
            Projection object and shape or a HEALPix instance.

        Raises
        ------
        ValueError
            If the projection format is invalid.
        """
        self.reset_cached_maps()

        if isinstance(projection, HEALPix):
            self._projection = projection
            npix = np.array(projection.npix, dtype=int)
            self._shape_out = (npix,)

        elif isinstance(projection, tuple) and isinstance(projection[0], WCS):
            self._projection = projection[0]
            self._shape_out = projection[1]

        else:
            raise ValueError(
                "Projection must be a HEALPix object or a tuple (WCS, array shape)."
            )

    def generate_simulation(self):
        """Generate simulated components.

        This method generates the simulated components required for the simulation. It uses the parameters supplied to the class to generate the components and stores them in the class instance.
        """
        sim_components = self.run_simulation(**self._simulation_parameters)
        for key, value in sim_components.items():
            setattr(self, f"_{key}_MAP", value)
            self.params_supplied.append(key)
        self._maps_cached = True

    def run_simulation(self) -> dict:
        """Template for method to generate simulated components required.

        Returns
        -------
        dictionary
            Dictionary of component shortforms (I,Q,U,V,P,A,PF) and arrays corresponding to them

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # ==========================================================================
    # Stokes component properties

    def _get_stokes_property(self, stokes_key: str):
        """Get the Stokes property for the given key.

        Parameters
        ----------
        stokes_key : str
            The Stokes parameter key (I, Q, U, V, P, A, PF) to retrieve.

        Returns
        -------
        numpy.ndarray
            The Stokes parameter map corresponding to the given key.
        """
        if not self._maps_cached:
            self.generate_simulation()
        func = get_stokes_value_mapping(
            stokes_key, self.params_supplied, assume_v_0=self.assume_v_0
        )
        return func(
            **{param: getattr(self, f"_{param}_MAP") for param in self.params_supplied}
        )

    @property
    def I(self):  # noqa: E743
        """Stokes I map.

        Returns
        -------
        numpy array
            Stokes I map data.
        """
        return self._get_stokes_property("I")

    @property
    def Q(self):
        """Stokes Q map.

        Returns
        -------
        numpy array
            Stokes Q map data.
        """
        return self._get_stokes_property("Q")

    @property
    def U(self):
        """Stokes U map.

        Returns
        -------
        numpy array
            Stokes U map data.
        """
        return self._get_stokes_property("U")

    @property
    def V(self):
        """Stokes V map.

        Returns
        -------
        numpy array
            Stokes V map data.
        """
        return self._get_stokes_property("V")

    @property
    def P(self):
        """Total polarization map.

        Returns
        -------
        numpy array
            Total polarization map data.
        """
        return self._get_stokes_property("P")

    @property
    def A(self):
        """Polarization angle map.

        Returns
        -------
        numpy array
            Polarization angle map data.
        """
        return self._get_stokes_property("A")

    @property
    def PF(self):
        """Polarization fraction map.

        Returns
        -------
        numpy array
            Polarization fraction map data.
        """
        return self._get_stokes_property("PF")


class stokesMapSimulation(stokesMap):
    """Class to hold simulated astronomical maps using the Stokes parameter convention in a mutable/editable manner."""

    def __init__(self, *args, assume_v_0=True, **kwargs):
        """Initialize stokesMap object."""
        super().__init__(*args, **kwargs)
        self.assume_v_0 = assume_v_0
        self.simulation_components = []

    def add_simulation_component(self, component):
        """Add a simulation component to the list of components."""
        if not isinstance(component, stokesMapSimulationComponent):
            raise TypeError(
                "component must be an instance of stokesMapSimulationComponent"
            )
        self.simulation_components.append(component)
        self.update_component_projections()

    def update_component_projections(self, proj=None, map_shape=None, **kwargs):
        """Update the projection for all simulation components."""
        if proj is None:
            for component in self.simulation_components:
                if isinstance(self.projection, HEALPix):
                    component.projection = self.projection
                elif isinstance(self.projection, WCS):
                    component.projection = (self.projection, self.shape)
        else:
            if isinstance(proj, HEALPix):
                shape = (proj.npix,)
            elif isinstance(proj, tuple) and isinstance(proj[0], WCS):
                shape = proj[1]
                proj = proj[0]
            for component in self.simulation_components:
                component.projection = proj
                component.shape = shape

    @property
    def projection(self):
        """Get the projection object for the simulation."""
        return self._projection

    @projection.setter
    def projection(self, proj):
        """Set the projection object for the map.

        Can accept either a WCS object or HEALPix object.

        Parameters
        ----------
        proj : WCS or HEALPix
            The projection object associated with the map.
        """
        if isinstance(proj, (HEALPix, WCS)):
            self._projection = proj
        else:
            raise ValueError("Projection must be either a WCS or HEALPix object.")

    def set_projection(self, proj):
        """Set the projection object for the simulation output, update all components and call set_projection."""
        if isinstance(proj, HEALPix):
            self.projection = proj
            self.shape = (proj.npix,)
        elif isinstance(proj, tuple) and isinstance(proj[0], WCS):
            self.projection = proj[0]
            self.shape = proj[1]
        else:
            raise ValueError(
                "Projection must be a HEALPix object or a tuple (WCS, array shape)."
            )
        self.update_component_projections()

    @property
    def shape(self):
        """Shape of the map data array."""
        return self._shape

    @shape.setter
    def shape(self, shp):
        """Set the shape of the map data array."""
        self._shape = np.array(shp)

    def _get_stokes_map(self, stokes_type):
        """Generate and combine the specified Stokes map from all simulation components.

        Parameters
        ----------
        stokes_type : str
            The type of Stokes parameter to generate (e.g., 'I', 'Q', 'U', 'V', 'P', 'A', 'PF').

        Returns
        -------
        numpy.ndarray
            Combined simulated Stokes map.

        Raises
        ------
        ValueError
            If the stokes_type is not valid.
        """
        if stokes_type not in {"I", "Q", "U", "V", "P", "A", "PF"}:
            raise ValueError(f"Invalid Stokes type: {stokes_type}")

        combined_map = None

        if stokes_type in {"I", "Q", "U", "V"}:
            for component in self.simulation_components:
                if combined_map is None:
                    combined_map = np.array(getattr(component, stokes_type), copy=True)
                else:
                    combined_map += np.array(getattr(component, stokes_type), copy=True)

        elif stokes_type == "P":
            for component in self.simulation_components:
                if combined_map is None:
                    combined_Q = np.array(getattr(component, "Q"), copy=True)
                    combined_U = np.array(getattr(component, "U"), copy=True)
                    combined_V = np.array(getattr(component, "V"), copy=True)
                else:
                    combined_Q += np.array(getattr(component, "Q"), copy=True)
                    combined_U += np.array(getattr(component, "U"), copy=True)
                    combined_V += np.array(getattr(component, "V"), copy=True)
            combined_map = np.sqrt(combined_Q**2 + combined_U**2 + combined_V**2)

        elif stokes_type == "A":
            for component in self.simulation_components:
                if combined_map is None:
                    combined_Q = np.array(getattr(component, "Q"), copy=True)
                    combined_U = np.array(getattr(component, "U"), copy=True)
                else:
                    combined_Q += np.array(getattr(component, "Q"), copy=True)
                    combined_U += np.array(getattr(component, "U"), copy=True)
            combined_map = 0.5 * np.arctan2(combined_U, combined_Q)
            combined_map = wrap_theta(combined_map)

        elif stokes_type == "PF":
            for component in self.simulation_components:
                if combined_map is None:
                    combined_P = np.array(getattr(component, "P"), copy=True)
                else:
                    combined_P += np.array(getattr(component, "P"), copy=True)
            combined_map = (combined_P / np.sqrt(combined_Q**2 + combined_U**2)) * 100

        return combined_map
