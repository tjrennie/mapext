"""Provide utilities for working with Stokes parameters.

Include classes and functions to compute, map, and derive Stokes parameters
such as I, Q, U, V, P, A, and PF, which are commonly used in polarimetry.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "StokesComp",
    "get_derivative_function",
    "get_stokes_value_mapping",
    "int_to_stokes",
    "queryable_parameters",
    "stokes_to_int",
    "wrap_theta",
]

display_parameters = {
    "I": r"$I$",
    "Q": r"$Q$",
    "U": r"$U$",
    "V": r"$V$",  # Standard Stokes parameters
    "P": r"$P$",  # Total polarisation
    "A": r"$\gamma$",  # Polarisation angle
    "PF": r"$p$",  # Total polarisation fraction
}

stokes_to_int: dict[str, int] = {
    "I": 0,
    "Q": 1,
    "U": 2,
    "V": 3,
    "P": 4,
    "A": 5,  # Also called 'theta'
    "PF": 6,
}

int_to_stokes: dict[int, str] = {v: k for k, v in stokes_to_int.items()}

queryable_parameters = list(display_parameters.keys())

_hpi = np.pi / 2


def wrap_theta(value: float) -> float:
    """Wrap the given angle value to the range [0, π].

    Parameters
    ----------
    value : float
        The angle value to be wrapped.

    Returns
    -------
    float
        The wrapped angle value within the range [0, π].
    """
    return value % np.pi


class StokesComp:
    """Represents a Stokes parameter component.

    This class provides methods to convert a Stokes parameter letter
    (e.g., 'I', 'Q', 'U', etc.) into its corresponding integer or float
    representation and vice versa.

    Attributes
    ----------
    letter : str
        The Stokes parameter letter (e.g., 'I', 'Q', 'U', etc.).

    Methods
    -------
    to_int() -> int
        Convert the Stokes parameter letter to its integer representation.
    to_float() -> float
        Convert the Stokes parameter letter to its float representation.
    """

    def __init__(self, letter: str):
        letter = letter.upper().strip()  # Normalize input
        if letter not in stokes_to_int:
            logger.error(f"Invalid Stokes parameter: {letter}")
            raise ValueError(f"Invalid parameter: {letter}")
        self.letter = letter

    def to_int(self) -> int:
        """Convert the Stokes parameter letter to its integer representation."""
        return stokes_to_int[self.letter]

    def to_float(self) -> float:
        """Convert the Stokes parameter letter to its float representation."""
        return float(stokes_to_int[self.letter])

    @classmethod
    def from_int(cls, value: int) -> "StokesComp":
        """Create a StokesComp from its integer representation."""
        if value not in int_to_stokes:
            logger.error(f"Invalid Stokes integer: {value}")
            raise ValueError(f"Invalid Stokes integer: {value}")
        return cls(int_to_stokes[value])

    def __repr__(self) -> str:
        """Convert the Stokes parameter letter to its string representation."""
        return f"Stokes('{self.letter}')"

    def __int__(self) -> int:
        """Convert the Stokes parameter letter to its integer representation."""
        return self.to_int()

    def __float__(self) -> float:
        """Convert the Stokes parameter letter to its float representation."""
        return self.to_float()

    def __eq__(self, other) -> bool:
        if isinstance(other, StokesComp):
            return self.letter == other.letter
        if isinstance(other, str):
            return self.letter == other.upper().strip()
        if isinstance(other, int):
            return self.to_int() == other
        return False

    def __hash__(self) -> int:
        return hash(self.letter)


def get_stokes_value_mapping(
    required, supplied, assume_v_0=True, use_derived_params=True, derivative=False
):
    """Get the Stokes mapping for the given parameters and return a function that takes a dictionary of values to map from.

    Parameters
    ----------
    required : str
        The required Stokes parameter to compute (e.g., 'I', 'Q', 'U', 'V', 'P', 'A', 'PF').
    supplied : list
        The list of supplied Stokes parameters (e.g., ['I', 'Q', 'U']).
    assume_v_0 : bool, optional
        Whether to assume V=0 if not supplied (default is True).
    use_derived_params : bool, optional
        Whether to use derived parameters (default is True).
    derivative : bool, optional
        Whether to compute the derivative of the mapping function (default is False).

    Returns
    -------
    function
        A function that takes a dictionary of values and returns the mapped value.
    """
    if not use_derived_params:
        supplied = [p for p in supplied if p not in ["P", "A", "PF"]]

    mapping_functions = {
        "I": get_I_mapping(),
        "Q": get_Q_mapping(assume_v_0),
        "U": get_U_mapping(assume_v_0),
        "V": get_V_mapping(assume_v_0),
        "P": get_P_mapping(assume_v_0),
        "A": get_A_mapping(),
        "PF": get_PF_mapping(assume_v_0),
    }

    if required.upper() not in mapping_functions:
        logger.error(f"Unknown required parameter: {required}")
        raise ValueError(f"Cannot compute '{required}' with the supplied parameters.")

    logger.info(f"Providing mapping function for Stokes parameter '{required.upper()}'")
    return get_derivative_function(mapping_functions[required.upper()], derivative)


def get_derivative_function(func, derivative):
    """Return the derivative function if requested, else the original."""
    return get_derivative(func) if derivative else func


def get_derivative(func):
    """Return a function to numerically approximate the derivative of the given function."""

    def derivative_func(**kwargs):
        logger.debug(f"Computing numerical derivative for: {kwargs}")
        epsilon = 1e-6
        params = kwargs.copy()
        derivatives = {}

        for param in kwargs:
            params[param] = kwargs[param] + epsilon
            f_plus = func(**params)
            params[param] = kwargs[param] - epsilon
            f_minus = func(**params)
            derivatives[param] = (f_plus - f_minus) / (2 * epsilon)

        return derivatives

    return derivative_func


def get_I_mapping():
    """Return a function to determine the Stokes I parameter."""

    def I_func(**kwargs):
        if "I" in kwargs:
            return kwargs["I"]
        raise ValueError("Missing 'I' parameter.")

    return I_func


def get_Q_mapping(assume_v_0=True):
    """Return a function to determine the Stokes Q parameter."""

    def Q_func(**kwargs):
        V = kwargs.get("V", 0 if assume_v_0 else None)

        if "Q" in kwargs:
            return kwargs["Q"]
        if "U" in kwargs and "A" in kwargs:
            return kwargs["U"] * np.tan(2 * kwargs["A"])
        if "P" in kwargs and "U" in kwargs and V is not None:
            return np.sqrt(kwargs["P"] ** 2 - (kwargs["U"] ** 2 + V**2))
        if "PF" in kwargs and "I" in kwargs and "U" in kwargs and V is not None:
            return np.sqrt(
                (kwargs["PF"] * kwargs["I"]) ** 2 - (kwargs["U"] ** 2 + V**2)
            )
        if "PF" in kwargs and "I" in kwargs and "A" in kwargs and np.all(V == 0):
            return (
                kwargs["I"] * kwargs["PF"] * np.sqrt(1 - (np.sin(2 * kwargs["A"])) ** 2)
            )

        logger.warning(f"Failed to compute Q with parameters: {kwargs}")
        raise ValueError(
            f"Cannot compute 'Q' with the supplied parameters {kwargs.keys()}."
        )

    return Q_func


def get_U_mapping(assume_v_0=True):
    """Return a function to determine the Stokes U parameter."""

    def U_func(**kwargs):
        V = kwargs.get("V", 0 if assume_v_0 else None)

        if "U" in kwargs:
            return kwargs["U"]
        if "Q" in kwargs and "A" in kwargs:
            return kwargs["Q"] / np.tan(2 * kwargs["A"])
        if "P" in kwargs and "Q" in kwargs and V is not None:
            return np.sqrt(kwargs["P"] ** 2 - (kwargs["Q"] ** 2 + V**2))
        if "PF" in kwargs and "I" in kwargs and "Q" in kwargs and V is not None:
            return np.sqrt(
                (kwargs["PF"] * kwargs["I"]) ** 2 - (kwargs["Q"] ** 2 + V**2)
            )
        if "PF" in kwargs and "I" in kwargs and "A" in kwargs and np.all(V == 0):
            return (
                kwargs["I"] * kwargs["PF"] * np.sqrt(1 - (np.cos(2 * kwargs["A"])) ** 2)
            )

        logger.warning(f"Failed to compute U with parameters: {kwargs}")
        raise ValueError(
            f"Cannot compute 'U' with the supplied parameters {kwargs.keys()}."
        )

    return U_func


def get_V_mapping(assume_v_0=True):
    """Return a function to determine the Stokes V parameter."""

    def V_func(**kwargs):
        if "V" in kwargs:
            return kwargs["V"]
        if assume_v_0:
            logger.debug("Assuming V=0 since it was not provided.")
            return 0
        logger.warning("V not supplied and assume_v_0 is False.")
        raise ValueError(
            f"Cannot compute 'V' with the supplied parameters {kwargs.keys()}."
        )

    return V_func


def get_P_mapping(assume_v_0=True):
    """Return a function to determine the total polarisation P parameter."""

    def P_func(**kwargs):
        V = kwargs.get("V", 0 if assume_v_0 else None)

        if "P" in kwargs:
            return kwargs["P"]
        if "Q" in kwargs and "U" in kwargs and V is not None:
            return np.sqrt(kwargs["Q"] ** 2 + kwargs["U"] ** 2 + V**2)
        if "PF" in kwargs and "I" in kwargs:
            return kwargs["PF"] * kwargs["I"]

        logger.warning(f"Failed to compute P with parameters: {kwargs}")
        raise ValueError(
            f"Cannot compute 'P' with the supplied parameters {kwargs.keys()}."
        )

    return P_func


def get_A_mapping():
    """Return a function to determine the polarisation angle A."""

    def A_func(**kwargs):
        if "A" in kwargs:
            return kwargs["A"]
        if "Q" in kwargs and "U" in kwargs:
            if kwargs["Q"] == 0 and kwargs["U"] == 0:
                return np.nan
            return np.arctan2(kwargs["U"], kwargs["Q"]) / 2

        logger.warning(f"Failed to compute A with parameters: {kwargs}")
        raise ValueError(
            f"Cannot compute 'A' with the supplied parameters {kwargs.keys()}."
        )

    return A_func


def get_PF_mapping(assume_v_0=True):
    """Return a function to determine the polarisation fraction PF."""

    def PF_func(**kwargs):
        V = kwargs.get("V", 0 if assume_v_0 else None)

        if "PF" in kwargs:
            return kwargs["PF"]
        if "I" in kwargs and "P" in kwargs:
            return kwargs["P"] / kwargs["I"]
        if "I" in kwargs and "Q" in kwargs and "U" in kwargs and V is not None:
            return np.sqrt(kwargs["Q"] ** 2 + kwargs["U"] ** 2 + V**2) / kwargs["I"]

        logger.warning(f"Failed to compute PF with parameters: {kwargs}")
        raise ValueError(
            f"Cannot compute 'PF' with the supplied parameters {kwargs.keys()}."
        )

    return PF_func
