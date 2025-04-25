from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mapext")
except PackageNotFoundError:
    # package not installed
    pass

del version, PackageNotFoundError
