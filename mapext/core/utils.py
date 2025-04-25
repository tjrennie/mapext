"""Utility functions for loading various file formats."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import yaml
from astropy.io import fits


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
    with fits.open(file) as hdul:
        # Use the first HDU with data
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data
                break
        else:
            return None  # No HDU had data

        array = np.array(data)

        # If all elements are strings, use chararray
        if array.dtype.kind in {"U", "S"}:
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
    df = pd.read_csv(file)
    array = df.to_numpy()

    # If all elements are strings, use chararray
    if array.dtype == object and all(isinstance(x, str) for x in array.flat):
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
    with open(file, encoding="utf-8") as f:
        return json.load(f)


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
    with open(file, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    with open(file, encoding="utf-8") as f:
        return toml.load(f)


DICTLIKE_LOADERS = {
    ".json": load_json,
    ".jsn": load_json,
    ".yaml": load_yaml,
    ".yml": load_yaml,
    ".toml": load_toml,
    ".tml": load_toml,
}


def autoload_dictlike(file: str):
    """Automatically load dictionary-like data (e.g., JSON, YAML, TOML)."""
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"File {file} does not exist.")

    loader = DICTLIKE_LOADERS.get(path.suffix.lower())
    if not loader:
        supported = ", ".join(DICTLIKE_LOADERS.keys())
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Supported types for dict-like loading: {supported}"
        )

    return loader(file)


ARRAYLIKE_LOADERS = {
    ".csv": load_csv,
    ".fits": load_fits,
    ".fit": load_fits,
}


def autoload_arraylike(file: str):
    """Automatically load array-like data (e.g., CSV, FITS) as plain NumPy arrays."""
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"File {file} does not exist.")

    loader = ARRAYLIKE_LOADERS.get(path.suffix.lower())
    if not loader:
        supported = ", ".join(ARRAYLIKE_LOADERS.keys())
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Supported types for array-like loading: {supported}"
        )

    return loader(file)
