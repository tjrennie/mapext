from . import core
from . import emission
from . import simulation

__all__ = ["core", "emission", "simulation", "__version__"]

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("mapext")
except PackageNotFoundError:
    logger.warning("Could not determine package version. Is 'mapext' installed?")
    __version__ = "unknown"

del version, PackageNotFoundError
