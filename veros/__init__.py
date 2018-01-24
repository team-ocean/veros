from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# for some reason, netCDF4 has to be imported before h5py, so we do it here
import netCDF4
import h5py

del netCDF4
del h5py

from .decorators import veros_method, veros_inline_method, veros_class_method
from .veros import Veros
from .veros_legacy import VerosLegacy
