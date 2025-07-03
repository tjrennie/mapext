"""Utility functions for loading various file formats."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import yaml
from astropy.io import fits
from astropy.units import Quantity, Unit

logger = logging.getLogger(__name__)


def load_fits(file):
    """Load a FITS file and return the data as a NumPy array.

    Parameters
    ----------
    file : str
        Path to the FITS file.

    Returns
    -------
    np.ndarray
        Data from the FITS file as a NumPy array. If the data is a string, it will be returned as a chararray.
    """
    logger.debug(f"Loading FITS file: {file}")
    with fits.open(file) as hdul:
        # Use the first HDU with data
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data
                logger.debug("Found HDU with data")
                break
        else:
            logger.warning("No HDU with data found")
            return None  # No HDU had data

        array = np.array(data)
        logger.debug(f"Data loaded, dtype={array.dtype}")

        # If all elements are strings, use chararray
        if array.dtype.kind in {"U", "S"}:
            logger.debug("Data is string-like, converting to chararray")
            return np.char.array(array)

        return array


def load_csv(file):
    """Load a CSV file and return the data as a NumPy array.

    Parameters
    ----------
    file : str
        Path to the CSV file.

    Returns
    -------
    np.ndarray
        Data from the CSV file as a NumPy array. If the data is a string, it will be returned as a chararray.
    """
    logger.debug(f"Loading CSV file: {file}")
    df = pd.read_csv(file)
    array = df.to_numpy()
    logger.debug(f"CSV loaded, shape={array.shape}, dtype={array.dtype}")

    # If all elements are strings, use chararray
    if array.dtype == object and all(isinstance(x, str) for x in array.flat):
        logger.debug("Data is string-like, converting to chararray")
        return np.char.array(array)

    return array


def load_json(file):
    """Load a JSON file and return the data as a dictionary.

    Parameters
    ----------
    file : str
        Path to the JSON file.

    Returns
    -------
    dict
        Data from the JSON file as a dictionary.
    """
    logger.debug(f"Loading JSON file: {file}")
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
    logger.debug("JSON loaded successfully")
    return data


def load_yaml(file):
    """Load a YAML file and return the data as a dictionary.

    Parameters
    ----------
    file : str
        Path to the YAML file.

    Returns
    -------
    dict
        Data from the YAML file as a dictionary.
    """
    logger.debug(f"Loading YAML file: {file}")
    with open(file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.debug("YAML loaded successfully")
    return data


def load_toml(file):
    """Load a TOML file and return the data as a dictionary.

    Parameters
    ----------
    file : str
        Path to the TOML file.

    Returns
    -------
    dict
        Data from the TOML file as a dictionary.
    """
    logger.debug(f"Loading TOML file: {file}")
    with open(file, encoding="utf-8") as f:
        data = toml.load(f)
    logger.debug("TOML loaded successfully")
    return data


DICTLIKE_LOADERS = {
    ".json": load_json,
    ".jsn": load_json,
    ".yaml": load_yaml,
    ".yml": load_yaml,
    ".toml": load_toml,
    ".tml": load_toml,
}


def autoload_dictlike(file: str):
    """Automatically load dictionary-like data (e.g., JSON, YAML, TOML).

    Parameters
    ----------
    file : str
        Path to the file to be loaded.

    Returns
    -------
    dict
        Data from the file as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file extension is not supported for dict-like loading.
    TypeError
        If the file is not a string or Path object.
    """
    logger.debug(f"Auto-loading dict-like file: {file}")
    path = Path(file)
    if not path.exists():
        logger.error(f"File not found: {file}")
        raise FileNotFoundError(f"File {file} does not exist.")

    loader = DICTLIKE_LOADERS.get(path.suffix.lower())
    if not loader:
        supported = ", ".join(DICTLIKE_LOADERS.keys())
        logger.error(f"Unsupported file extension for dict-like loading: {path.suffix}")
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Supported types for dict-like loading: {supported}"
        )

    data = loader(file)
    logger.debug(f"File loaded successfully with loader for {path.suffix}")
    return data


ARRAYLIKE_LOADERS = {
    ".csv": load_csv,
    ".fits": load_fits,
    ".fit": load_fits,
}


def autoload_arraylike(file: str):
    """Automatically load array-like data (e.g., CSV, FITS) as plain NumPy arrays.

    Parameters
    ----------
    file : str
        Path to the file to be loaded.

    Returns
    -------
    np.ndarray
        Data from the file as a NumPy array. If the data is a string, it will be returned as a chararray.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file extension is not supported for array-like loading.
    TypeError
        If the file is not a string or Path object.
    """
    logger.debug(f"Auto-loading array-like file: {file}")
    path = Path(file)
    if not path.exists():
        logger.error(f"File not found: {file}")
        raise FileNotFoundError(f"File {file} does not exist.")

    loader = ARRAYLIKE_LOADERS.get(path.suffix.lower())
    if not loader:
        supported = ", ".join(ARRAYLIKE_LOADERS.keys())
        logger.error(
            f"Unsupported file extension for array-like loading: {path.suffix}"
        )
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Supported types for array-like loading: {supported}"
        )

    data = loader(file)
    logger.debug(f"File loaded successfully with loader for {path.suffix}")
    return data


def string_to_astropy_quantity(quantity_string):
    """Convert a string or Quantity to an astropy Quantity object.

    Parameters
    ----------
    quantity_string : str or Quantity
        The input quantity to convert.

    Returns
    -------
    Quantity
        The converted astropy Quantity object.
    """
    if isinstance(quantity_string, str):
        return Quantity(quantity_string)
    if isinstance(quantity_string, Quantity):
        return quantity_string
    raise ValueError('Cannot convert input to astropy Quantity object.')

def string_to_astropy_unit(unit_string):
    """Convert a string or Unit to an astropy Unit object.

    Parameters
    ----------
    unit_string : str or Unit
        The input unit to convert.

    Returns
    -------
    Unit
        The converted astropy Unit object.
    """
    if isinstance(unit_string, str):
        return Unit(unit_string)
    if isinstance(unit_string, Unit):
        return unit_string
    raise ValueError('Cannot convert input to astropy Unit object.')
